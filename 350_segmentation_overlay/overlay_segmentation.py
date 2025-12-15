
import numpy as np

def main():
    try:
        from PIL import Image
    except Exception:
        print('Pillow not available; cannot save overlay image.')
        return
    h, w = 256, 256
    img = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(h, w), dtype=np.uint8)
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]
    out = Image.fromarray(overlay)
    out.save('segmentation_overlay.png')
    print('Saved segmentation_overlay.png')

if __name__ == '__main__':
    main()
