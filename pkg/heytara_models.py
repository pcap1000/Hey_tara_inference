"""Wake-word detector architectures. Every model maps a standardised log-mel batch (B, 1, n_mels, T) to one logit per clip.

tc_resnet   - temporal-convolution ResNet (mel bins are channels)                      [Choi et al. 2019]
crnn        - 2D conv front-end + bidirectional GRU
ds_cnn      - depthwise-separable CNN ("Hello Edge")                                   [Zhang et al. 2017]
bc_resnet   - broadcasted-residual network, BC-ResNet-tau                              [Kim et al. 2021]
matchboxnet - 1D time-channel-separable convolutions, MatchboxNet-3x1x64               [Majumdar & Ginsburg 2020]
micronet    - MicroNet-STYLE depthwise-separable net: non-uniform widths, ReLU6, int8-friendly design.
              This is an approximation of the MicroNet-KWS design (Banbury et al. 2021), NOT the published searched
              architecture or weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, k=3, s=1, p=None, g=1, act=nn.ReLU):
    p = k // 2 if p is None else p
    layers = [nn.Conv2d(i, o, k, s, p, groups=g, bias=False), nn.BatchNorm2d(o)]
    if act is not None:
        layers.append(act(inplace=True))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ TC-ResNet
class TCBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.a = nn.Sequential(nn.Conv1d(i, o, 9, 2, 4, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True),
                            nn.Conv1d(o, o, 9, 1, 4, bias=False), nn.BatchNorm1d(o))
        s.sk = nn.Sequential(nn.Conv1d(i, o, 1, 2, bias=False), nn.BatchNorm1d(o))

    def forward(s, x):
        return F.relu(s.a(x) + s.sk(x))


class TCResNet(nn.Module):
    def __init__(s, n_mels, ch=(48, 64, 96, 128)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(n_mels, ch[0], 3, 1, 1, bias=False), nn.BatchNorm1d(ch[0]), nn.ReLU(inplace=True))
        s.b = nn.Sequential(*[TCBlock(a, b) for a, b in zip(ch[:-1], ch[1:])])
        s.h = nn.Linear(ch[-1], 1)

    def forward(s, x):
        return s.h(s.b(s.stem(x[:, 0])).mean(-1)).squeeze(1)


# ------------------------------------------------------------------ CRNN
class CRNN(nn.Module):
    def __init__(s, n_mels, hid=96):
        super().__init__()
        s.f = nn.Sequential(cbr(1, 32), nn.MaxPool2d((2, 1)), cbr(32, 48), nn.MaxPool2d((2, 2)), cbr(48, 64), nn.MaxPool2d((2, 1)))
        s.rnn = nn.GRU(64 * (n_mels // 8), hid, batch_first=True, bidirectional=True)
        s.h = nn.Linear(4 * hid, 1)

    def forward(s, x):
        z = s.f(x).permute(0, 3, 1, 2).flatten(2)
        o, _ = s.rnn(z)
        return s.h(torch.cat([o.mean(1), o.max(1).values], 1)).squeeze(1)


# ------------------------------------------------------------------ DS-CNN
class DSCNN(nn.Module):
    def __init__(s, n_mels, c=64, n=5):
        super().__init__()
        s.stem = cbr(1, c, (10, 4), (2, 2), (5, 2))
        s.b = nn.Sequential(*[nn.Sequential(cbr(c, c, 3, 1, 1, g=c), cbr(c, c, 1, 1, 0)) for _ in range(n)])
        s.drop, s.h = nn.Dropout(0.2), nn.Linear(c, 1)

    def forward(s, x):
        return s.h(s.drop(s.b(s.stem(x)).mean((2, 3)))).squeeze(1)


# ------------------------------------------------------------------ BC-ResNet
class BCResBlock(nn.Module):
    """Broadcasted residual block: a frequency-wise depthwise conv (f2), then a temporal branch (f1) computed on the
    frequency-averaged features and broadcast back over frequency; identity shortcut when channels are unchanged."""

    def __init__(s, cin, cout, stride_f=1, dil_t=1):
        super().__init__()
        s.transition = cin != cout
        s.pre = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)) if s.transition else nn.Identity()
        s.f2 = nn.Sequential(nn.Conv2d(cout, cout, (3, 1), (stride_f, 1), (1, 0), groups=cout, bias=False), nn.BatchNorm2d(cout))
        s.f1 = nn.Sequential(nn.Conv2d(cout, cout, (1, 3), 1, (0, dil_t), dilation=(1, dil_t), groups=cout, bias=False),
                             nn.BatchNorm2d(cout), nn.SiLU(inplace=True), nn.Conv2d(cout, cout, 1, bias=False), nn.Dropout2d(0.1))

    def forward(s, x):
        x = s.pre(x)
        sc = x
        a = s.f2(x)                               # (B, C, F', T)
        b = s.f1(a.mean(2, keepdim=True))         # (B, C, 1, T), broadcast over frequency
        y = a + b
        if not s.transition:
            y = y + sc
        return F.relu(y)


class BCResNet(nn.Module):
    def __init__(s, n_mels, tau=3):
        super().__init__()
        c = [int(v * tau) for v in (8, 12, 16, 20)]
        n, strides, dils = (2, 2, 4, 4), (1, 2, 2, 1), (1, 2, 4, 8)
        s.stem = nn.Sequential(nn.Conv2d(1, 16 * tau, 5, (2, 1), 2, bias=False), nn.BatchNorm2d(16 * tau), nn.ReLU(inplace=True))
        blocks, cin = [], 16 * tau
        for cs, ns, st, d in zip(c, n, strides, dils):
            for j in range(ns):
                blocks.append(BCResBlock(cin, cs, st if j == 0 else 1, d))
                cin = cs
        s.blocks = nn.Sequential(*blocks)
        s.head = nn.Sequential(nn.Conv2d(cin, cin, 5, 1, 2, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                               nn.Conv2d(cin, 32 * tau, 1, bias=False), nn.BatchNorm2d(32 * tau), nn.ReLU(inplace=True))
        s.h = nn.Linear(32 * tau, 1)

    def forward(s, x):
        return s.h(s.head(s.blocks(s.stem(x))).mean((2, 3))).squeeze(1)


# ------------------------------------------------------------------ MatchboxNet
class MBBlock(nn.Module):
    def __init__(s, cin, cout, k, R=1, dropout=0.1):
        super().__init__()
        layers = []
        for r in range(R):
            i = cin if r == 0 else cout
            layers += [nn.Conv1d(i, i, k, 1, k // 2, groups=i, bias=False), nn.Conv1d(i, cout, 1, bias=False), nn.BatchNorm1d(cout)]
            if r < R - 1:
                layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
        s.sub = nn.Sequential(*layers)
        s.res = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
        s.out = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(s, x):
        return s.out(s.sub(x) + s.res(x))


class MatchboxNet(nn.Module):
    """MatchboxNet-3x1x64: prologue (k=11, stride 2), 3 blocks (k=13, 15, 17), epilogue (k=29 dilated) and a 1x1 head."""

    def __init__(s, n_mels, c=64, kernels=(13, 15, 17), R=1):
        super().__init__()
        s.pro = nn.Sequential(nn.Conv1d(n_mels, 128, 11, 2, 5, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        blocks, cin = [], 128
        for k in kernels:
            blocks.append(MBBlock(cin, c, k, R)); cin = c
        s.blocks = nn.Sequential(*blocks)
        s.epi = nn.Sequential(nn.Conv1d(cin, cin, 29, 1, 28, dilation=2, groups=cin, bias=False), nn.Conv1d(cin, 128, 1, bias=False),
                              nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        s.h = nn.Linear(128, 1)

    def forward(s, x):
        return s.h(s.epi(s.blocks(s.pro(x[:, 0]))).mean(-1)).squeeze(1)


# ------------------------------------------------------------------ MicroNet-style
class MicroNetStyle(nn.Module):
    """Depthwise-separable KWS network in the spirit of MicroNet-KWS: non-uniform widths (grow, then narrow), ReLU6 for
    quantisation-friendly activations, no large pooling stages, global average pool + one linear layer.
    APPROXIMATION - the published MicroNets use channel widths found by differentiable architecture search."""

    def __init__(s, n_mels, widths=(64, 96, 96, 128, 96, 64), stem=48, time_strides=(2, 1, 2, 1, 1, 1)):
        super().__init__()
        s.stem = cbr(1, stem, (10, 4), (2, 2), (5, 2), act=nn.ReLU6)
        blocks, cin = [], stem
        for j, (w, ts) in enumerate(zip(widths, time_strides)):
            k = 5 if j % 2 == 0 else 3
            blocks.append(nn.Sequential(cbr(cin, cin, k, (1, ts), k // 2, g=cin, act=nn.ReLU6), cbr(cin, w, 1, 1, 0, act=nn.ReLU6)))
            cin = w
        s.b = nn.Sequential(*blocks)
        s.h = nn.Linear(cin, 1)

    def forward(s, x):
        return s.h(s.b(s.stem(x)).mean((2, 3))).squeeze(1)


# ------------------------------------------------------------------ registry
REGISTRY = {                     # name -> (class, learning rate, one-line description)
    "tc_resnet": (TCResNet, 3e-3, "TC-ResNet"),
    "crnn": (CRNN, 2e-3, "Conv + BiGRU"),
    "ds_cnn": (DSCNN, 3e-3, "DS-CNN (Hello Edge)"),
    "bc_resnet": (BCResNet, 3e-3, "BC-ResNet-3"),
    "matchboxnet": (MatchboxNet, 2e-3, "MatchboxNet-3x1x64"),
    "micronet": (MicroNetStyle, 3e-3, "MicroNet-style (approximation)"),
}


def build(name, n_mels):
    return REGISTRY[name][0](n_mels)


class AutocastWrap(nn.Module):
    """Runs the model in float16 autocast on CUDA and returns float32 logits. The autocast context is inside forward so
    it also applies to DataParallel replicas (which run in their own threads)."""

    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            return s.m(x).float()
