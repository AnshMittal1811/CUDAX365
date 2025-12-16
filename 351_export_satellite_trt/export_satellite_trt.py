
import numpy as np

def main():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print('Torch not available')
        return

    class TinySeg(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.head = nn.Conv2d(8, 1, kernel_size=1)
        def forward(self, x):
            x = torch.relu(self.conv(x))
            return self.head(x)

    model = TinySeg().eval()
    x = torch.randn(1, 3, 64, 64)
    onnx_path = 'tiny_seg.onnx'
    torch.onnx.export(model, x, onnx_path, input_names=['input'], output_names=['mask'], opset_version=13)
    print(f'Exported {onnx_path}')

    try:
        import tensorrt as trt
        print('TensorRT available; engine build can be added here.')
    except Exception as e:
        print('TensorRT not available:', e)

if __name__ == '__main__':
    main()
