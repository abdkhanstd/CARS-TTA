"""2D U-Net with instance normalisation (nnU-Net style), 4 output classes."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, n_cls=4, feats=(32, 64, 128, 256, 320)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = in_ch
        for f in feats:
            self.enc.append(Block(c, f))
            c = f
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for f in reversed(feats[:-1]):
            self.up.append(nn.ConvTranspose2d(c, f, 2, stride=2))
            self.dec.append(Block(2 * f, f))
            c = f
        self.head = nn.Conv2d(c, n_cls, 1)

    def forward(self, x, return_bottleneck=False):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        bott = x
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        out = self.head(x)
        if return_bottleneck:
            return out, bott
        return out


def norm_affine_params(model):
    """Parameters updated by test-time adaptation: affine weights and biases of all normalisation layers."""
    ps = []
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) and getattr(m, "weight", None) is not None:
            ps += [m.weight, m.bias]
    return ps
