
import time
import json
import numpy as np

def main():
    try:
        import fastvideo  # type: ignore
        has_fastvideo = True
    except Exception:
        has_fastvideo = False

    frames = 32
    h, w = 128, 128
    t0 = time.perf_counter()
    if has_fastvideo:
        # Placeholder: actual FastVideo API may differ.
        # We simulate by allocating dummy frames.
        _ = np.zeros((frames, h, w, 3), dtype=np.uint8)
    else:
        for _ in range(frames):
            _ = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    t1 = time.perf_counter()

    fps = frames / (t1 - t0)
    results = {'frames': frames, 'fps': fps, 'fastvideo_available': has_fastvideo}
    with open('fastvideo_example_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
