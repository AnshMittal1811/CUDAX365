import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        batch, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.weights.shape[1], h, w//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bchw,cohw->bohw", x_ft[:, :, :self.modes, :self.modes], self.weights
        )
        x = torch.fft.irfft2(out_ft, s=(h, w))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.conv0 = SpectralConv2d(width, width, modes)
        self.w0 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = torch.relu(x1 + x2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FNO2d().to(device)
    x = torch.randn(4, 64, 64, 1, device=device)
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main()
