import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_size=16, n_batches=10):
        super().__init__()
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.current = 0
        self.host_data = np.random.randn(batch_size, 128).astype(np.float32)
        self.device_input = cuda.mem_alloc(self.host_data.nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= self.n_batches:
            return None
        cuda.memcpy_htod(self.device_input, self.host_data)
        self.current += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None


def build_engine(onnx_path, int8=True, fp8=False):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = RandomCalibrator()
    if fp8 and hasattr(trt.BuilderFlag, "FP8"):
        config.set_flag(trt.BuilderFlag.FP8)

    engine = builder.build_engine(network, config)
    return engine


def main():
    engine = build_engine("mlp.onnx", int8=True, fp8=False)
    if engine is None:
        raise SystemExit("Engine build failed")
    with open("mlp_int8.plan", "wb") as f:
        f.write(engine.serialize())
    print("wrote mlp_int8.plan")


if __name__ == "__main__":
    main()
