
import os
import time
import numpy as np

def build_torch_model():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    return model

def main():
    model = build_torch_model()
    if model is None:
        print('Torch not available; skipping TensorRT demo.')
        return

    import torch
    x = torch.randn(1, 128)
    onnx_path = 'tiny_model.onnx'
    torch.onnx.export(model, x, onnx_path, input_names=['input'], output_names=['output'], opset_version=13)
    print(f'Exported {onnx_path}')

    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print('Failed to parse ONNX with TensorRT')
                return
        config = builder.create_builder_config()
        engine = builder.build_engine(network, config)
        context = engine.create_execution_context()
        inp = torch.randn(1, 128).numpy().astype(np.float32)
        out = np.empty((1, 10), dtype=np.float32)
        d_in = cuda.mem_alloc(inp.nbytes)
        d_out = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(d_in, inp)
        context.execute_v2([int(d_in), int(d_out)])
        cuda.memcpy_dtoh(out, d_out)
        print('TensorRT output shape:', out.shape)
    except Exception as e:
        print('TensorRT not available or failed:', e)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            out = sess.run(None, {'input': x.numpy()})
            print('ONNXRuntime output shape:', out[0].shape)
        except Exception as e2:
            print('ONNXRuntime fallback failed:', e2)
            with torch.no_grad():
                y = model(x)
                print('Torch output shape:', y.shape)

if __name__ == '__main__':
    main()
