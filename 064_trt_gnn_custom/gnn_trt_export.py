import torch
import torch.nn as nn

class TinyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x)


def main():
    model = TinyGNN().eval()
    x = torch.randn(1, 8)
    torch.onnx.export(model, x, "gnn.onnx", input_names=["x"], output_names=["y"], opset_version=17)
    print("wrote gnn.onnx")


if __name__ == "__main__":
    main()
