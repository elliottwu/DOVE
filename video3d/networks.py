import numpy as np
import torch
import torch.nn as nn
import torchvision


EPS = 1e-7


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    else:
        raise NotImplementedError


class ParamAct(nn.Module):
    def __init__(self, size, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.activation = get_activation(activation) if activation is not None else None

    def forward(self):
        out = self.weight
        if self.activation is not None:
            out = self.activation(out)
        return out


class LookupTable(nn.Module):
    def __init__(self, size, activation=None):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(size))
        self.activation = get_activation(activation) if activation is not None else None

    def forward(self, seq_idx, frame_idx=None):
        out = self.table[seq_idx]  # Bx(Fx)...
        if frame_idx is not None:
            ## frame_idx: BxF
            for _ in range(out.dim()-2):
                frame_idx = frame_idx.unsqueeze(-1)
            frame_idx = frame_idx.repeat(1,1,*out.shape[2:])
            out = out.gather(1, frame_idx)
        if self.activation is not None:
            out = self.activation(out)
        return out  # Bx(Fx)...


class MLP(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        super().__init__()
        assert num_layers >= 1
        if num_layers == 1:
            network = [nn.Linear(cin, cout, bias=False)]
        else:
            network = [nn.Linear(cin, nf, bias=False)]
            for _ in range(num_layers-2):
                network += [
                    nn.ReLU(inplace=True),
                    nn.Linear(nf, nf, bias=False)]
                if dropout > 0:
                    network += [nn.Dropout(dropout)]
            network += [
                nn.ReLU(inplace=True),
                nn.Linear(nf, cout, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*8, nf*8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    nn.GroupNorm(16*8, nf*8),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if zdim is None:
            network += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class EncoderDecoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, out_size=128, zdim=128, nf=64, activation=None):
        super().__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*8, nf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    nn.GroupNorm(16*8, nf*8),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)
        ]

        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
        ]

        add_upsample = int(np.log2(out_size//64))
        if add_upsample > 0:
            for _ in range(add_upsample):
                network += [
                    nn.Upsample(scale_factor=2, mode='nearest'),  # 4x4 -> 8x8
                    nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(16*8, nf*8),
                    nn.ReLU(inplace=True),
                ]

        network += [
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8 -> 16x16
            nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
            nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)
