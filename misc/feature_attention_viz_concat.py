import argparse
import cv2
import numpy as np
import os

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def resize_to_same_height(images: list[np.ndarray]) -> list[np.ndarray]:
    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / float(h)
        new_w = int(round(w * scale))
        resized.append(cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA))
    return resized

def resize_to_same_width(images: list[np.ndarray]) -> list[np.ndarray]:
    target_w = min(img.shape[1] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_w / float(w)
        new_h = int(round(h * scale))
        resized.append(cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA))
    return resized

def main():
    parser = argparse.ArgumentParser(description="Concatenate 3 visualization images")
    parser.add_argument("--inputs", nargs=3, required=True, help="Three input image paths")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--orientation", choices=["h", "v"], default="h", help="h: horizontal, v: vertical")
    args = parser.parse_args()

    imgs = [read_image(p) for p in args.inputs]

    if args.orientation == "h":
        imgs = resize_to_same_height(imgs)
        concat = np.hstack(imgs)
    else:
        imgs = resize_to_same_width(imgs)
        concat = np.vstack(imgs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ok = cv2.imwrite(args.output, concat)
    if not ok:
        raise RuntimeError(f"Failed to write output to {args.output}")

if __name__ == "__main__":
    main()