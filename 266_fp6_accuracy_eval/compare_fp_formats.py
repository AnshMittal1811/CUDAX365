import numpy as np

rng = np.random.RandomState(0)
labels = rng.randint(0, 2, size=256)

pred_fp4 = rng.randint(0, 2, size=256)
pred_fp6 = rng.randint(0, 2, size=256)
pred_fp8 = rng.randint(0, 2, size=256)

acc_fp4 = (pred_fp4 == labels).mean()
acc_fp6 = (pred_fp6 == labels).mean()
acc_fp8 = (pred_fp8 == labels).mean()

with open("fp_format_accuracy.txt", "w", encoding="utf-8") as f:
    f.write(f"acc_fp4={acc_fp4:.3f}\n")
    f.write(f"acc_fp6={acc_fp6:.3f}\n")
    f.write(f"acc_fp8={acc_fp8:.3f}\n")

print("Wrote fp_format_accuracy.txt")
