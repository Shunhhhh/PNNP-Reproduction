import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NoiseEmbedding(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX = -13.8, -4.6
        self.LOG_SIGMA2_MIN, self.LOG_SIGMA2_MAX = -23.0, -5.3
        self.LOG_ROW_STD_MIN, self.LOG_ROW_STD_MAX = -18.4, -2.3

    def forward(self, alpha, sigma2, row_std=None):
        alpha = torch.clamp(alpha, min=1e-8)
        sigma2 = torch.clamp(sigma2, min=1e-12)
        if row_std is None:
            row_std = torch.zeros_like(alpha)
        row_std = torch.clamp(row_std, min=1e-8)

        log_alpha = torch.log(alpha).unsqueeze(-1)
        log_sigma2 = torch.log(sigma2).unsqueeze(-1)
        log_row_std = torch.log(row_std).unsqueeze(-1)

        log_alpha = 2 * (log_alpha - self.LOG_ALPHA_MIN) / (
            self.LOG_ALPHA_MAX - self.LOG_ALPHA_MIN
        ) - 1
        log_sigma2 = 2 * (log_sigma2 - self.LOG_SIGMA2_MIN) / (
            self.LOG_SIGMA2_MAX - self.LOG_SIGMA2_MIN
        ) - 1
        log_row_std = 2 * (log_row_std - self.LOG_ROW_STD_MIN) / (
            self.LOG_ROW_STD_MAX - self.LOG_ROW_STD_MIN
        ) - 1

        return self.mlp(torch.cat([log_alpha, log_sigma2, log_row_std], dim=-1))


class AdaLN2d(nn.Module):
    def __init__(self, c, embed_dim=256):
        super().__init__()
        self.norm = LayerNorm2d(c)
        self.proj = nn.Linear(embed_dim, 2 * c)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, noise_emb):
        x = self.norm(x)
        gamma_beta = self.proj(noise_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * x + beta


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, embed_dim=256):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.adaln1 = AdaLN2d(c, embed_dim)
        self.adaln2 = AdaLN2d(c, embed_dim)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, noise_emb):
        x = self.adaln1(inp, noise_emb)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.adaln2(y, noise_emb)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_channel=12,
        out_channel=4,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        embed_dim=256,
    ):
        super().__init__()
        self.out_channel = out_channel
        self.intro = nn.Conv2d(in_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.ModuleList([NAFBlock(chan, embed_dim=embed_dim) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.ModuleList([NAFBlock(chan, embed_dim=embed_dim) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.ModuleList([NAFBlock(chan, embed_dim=embed_dim) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp_cond, inp_raw, noise_emb):
        _, _, H, W = inp_raw.shape
        inp_cond = self.check_image_size(inp_cond)

        x = self.intro(inp_cond)
        encs = []
        for encoder_blks, down in zip(self.encoders, self.downs):
            for blk in encoder_blks:
                x = blk(x, noise_emb)
            encs.append(x)
            x = down(x)

        for blk in self.middle_blks:
            x = blk(x, noise_emb)

        for decoder_blks, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for blk in decoder_blks:
                x = blk(x, noise_emb)

        x = self.ending(x)
        x = x[:, :, :H, :W]
        return x + inp_raw

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


class ConditionalNAFNet(nn.Module):
    def __init__(
        self,
        img_channel=4,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        embed_dim=256,
    ):
        super().__init__()
        self.noise_embedding = NoiseEmbedding(embed_dim)
        self.nafnet = NAFNet(
            in_channel=img_channel * 2,
            out_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            embed_dim=embed_dim,
        )

    def forward(self, x, alpha, sigma2, row_std=None, var_map=None, row_profile=None):
        if var_map is None:
            var_map = alpha.view(-1, 1, 1, 1) * torch.clamp(x, min=0) + sigma2.view(-1, 1, 1, 1)
        if row_profile is None:
            row_profile = torch.zeros_like(x)
        if row_profile.shape[-1] == 1:
            row_profile = row_profile.expand(-1, -1, -1, x.shape[-1])

        x_cond = torch.cat([x, var_map], dim=1)
        noise_emb = self.noise_embedding(alpha, sigma2, row_std)
        return self.nafnet(x_cond, x, noise_emb)