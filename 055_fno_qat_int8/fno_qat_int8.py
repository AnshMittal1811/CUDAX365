import torch
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8, 1)
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.mean(dim=[2,3])
        return self.fc(x)


def main():
    model = SmallNet()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(model, inplace=True)
    x = torch.randn(4, 1, 32, 32)
    y = model(x)
    loss = y.mean()
    loss.backward()
    torch.quantization.convert(model, inplace=True)
    print("QAT done")


if __name__ == "__main__":
    main()
