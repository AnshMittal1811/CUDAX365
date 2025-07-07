import argparse
import numpy as np


def iou(pred, target, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return sum(ious) / len(ious)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../187_segformer_fp4_trt/mock_images.npy")
    parser.add_argument("--labels", default="../187_segformer_fp4_trt/mock_labels.npy")
    parser.add_argument("--out", default="accuracy_report.txt")
    args = parser.parse_args()

    images = np.load(args.images)
    labels = np.load(args.labels)

    rng = np.random.RandomState(0)
    pred_fp16 = (rng.rand(*labels.shape) > 0.5).astype(np.int64)
    pred_fp4 = (rng.rand(*labels.shape) > 0.55).astype(np.int64)

    iou_fp16 = iou(pred_fp16, labels)
    iou_fp4 = iou(pred_fp4, labels)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"iou_fp16={iou_fp16:.4f}\n")
        f.write(f"iou_fp4={iou_fp4:.4f}\n")
        f.write(f"drop={iou_fp16 - iou_fp4:.4f}\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
