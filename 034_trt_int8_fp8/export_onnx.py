import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    def forward(self, x):
        return self.net(x)

model = MLP().eval()
example = torch.randn(1, 128)

torch.onnx.export(model, example, "mlp.onnx", input_names=["x"], output_names=["y"], opset_version=17)
print("wrote mlp.onnx")
