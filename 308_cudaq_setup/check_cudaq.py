try:
    import cudaq
    cudaq.set_target("qpp")

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    result = cudaq.sample(bell)
    with open("cudaq_setup_log.txt", "w", encoding="utf-8") as f:
        f.write(str(result))
    print("Wrote cudaq_setup_log.txt")
except Exception:
    with open("cudaq_setup_log.txt", "w", encoding="utf-8") as f:
        f.write("cudaq not available\n")
    print("cudaq not available")
