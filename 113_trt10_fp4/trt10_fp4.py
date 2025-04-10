import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def main():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open("mlp.onnx", "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if hasattr(trt.BuilderFlag, "FP4"):
        config.set_flag(trt.BuilderFlag.FP4)
        print("FP4 flag enabled")
    else:
        print("FP4 flag not available; using FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise SystemExit("Engine build failed")

    with open("mlp_trt.plan", "wb") as f:
        f.write(engine.serialize())
    print("wrote mlp_trt.plan")


if __name__ == "__main__":
    main()
