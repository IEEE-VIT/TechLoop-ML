#!/usr/bin/env python3
"""
predict_digit_ts_mlp.py

Inference script for your MNIST MLP exported as TorchScript: mnist_digit_model_ts.pt

- Loads TorchScript model (no need to import your notebook class)
- Preprocesses a handwritten digit photo to look MNIST-like:
  autocontrast -> optional invert -> threshold -> bbox crop -> resize to fit 20x20
  -> paste into 28x28 -> center-of-mass recenter -> (optional slight blur)
- Predicts a single image or a folder of images
- Can save the final 28x28 input as debug_28x28.png so you can verify preprocessing

Examples:
  python predict_digit_ts_mlp.py --model mnist_digit_model_ts.pt --image my_digit.jpg --debug
  python predict_digit_ts_mlp.py --model mnist_digit_model_ts.pt --image my_digit.jpg --invert --threshold 140 --debug
  python predict_digit_ts_mlp.py --model mnist_digit_model_ts.pt --folder ./digits --recursive --invert --threshold 120
"""

import os
import glob
import argparse

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_images(folder, recursive=False):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    if recursive:
        for e in exts:
            yield from glob.glob(os.path.join(folder, "**", e), recursive=True)
    else:
        for e in exts:
            yield from glob.glob(os.path.join(folder, e))


def preprocess_mnist_like(
    path: str,
    invert: bool | None = None,
    threshold: int = 120,
    blur: float = 0.3,
    debug: bool = False,
    debug_out: str = "debug_28x28.png",
) -> torch.Tensor:
    """
    Returns: [1, 1, 28, 28] float tensor in [0,1] (matches your training: ToTensor only)
    """
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)

    # Decide invert: MNIST expects black bg + white digit
    if invert is None:
        avg = float(np.array(img).mean())
        do_invert = avg > 127.0
    else:
        do_invert = invert

    if do_invert:
        img = ImageOps.invert(img)

    arr = np.array(img)

    # Threshold mask to find digit region
    mask = (arr > threshold).astype(np.uint8)

    if mask.sum() < 10:
        # Fallback: just resize whole image
        digit28 = img.resize((28, 28), Image.BILINEAR)
    else:
        ys, xs = np.where(mask == 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        x0, x1 = int(xs.min()), int(xs.max() + 1)

        crop = arr[y0:y1, x0:x1]

        # Resize keeping aspect so longest side becomes 20
        h, w = crop.shape
        if h > w:
            new_h = 20
            new_w = max(1, int(round(w * (20 / h))))
        else:
            new_w = 20
            new_h = max(1, int(round(h * (20 / w))))

        crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.BILINEAR)

        # Paste into 28x28 centered
        canvas = Image.new("L", (28, 28), 0)
        left = (28 - new_w) // 2
        top = (28 - new_h) // 2
        canvas.paste(crop_img, (left, top))

        # Center-of-mass recentering
        c = np.array(canvas).astype(np.float32) / 255.0
        total = float(c.sum())
        if total > 1e-6:
            yy, xx = np.indices(c.shape)
            cy = float((yy * c).sum() / total)
            cx = float((xx * c).sum() / total)
            shift_y = int(round(14 - cy))
            shift_x = int(round(14 - cx))
            c = np.roll(c, shift_y, axis=0)
            c = np.roll(c, shift_x, axis=1)

        digit28 = Image.fromarray((c * 255.0).clip(0, 255).astype(np.uint8))

    if blur and blur > 0:
        digit28 = digit28.filter(ImageFilter.GaussianBlur(radius=float(blur)))

    if debug:
        digit28.save(debug_out)
        a = np.array(digit28)
        print(
            f"[DEBUG] saved {debug_out} | invert={do_invert} thr={threshold} blur={blur} | "
            f"min={a.min()} max={a.max()} mean={a.mean():.2f}"
        )

    # Tensor in [0,1]
    x = torch.from_numpy(np.array(digit28)).float() / 255.0  # [28,28]
    x = x.unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return x


@torch.no_grad()
def predict(ts_model, x, device):
    ts_model.eval()
    logits = ts_model(x.to(device))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(probs.argmax().item())
    conf = float(probs[pred].item())
    return pred, conf, probs.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mnist_digit_model_ts.pt", help="TorchScript model .pt")
    ap.add_argument("--image", default=None, help="Single image path")
    ap.add_argument("--folder", default=None, help="Folder of images")
    ap.add_argument("--recursive", action="store_true", help="Search folder recursively")
    ap.add_argument("--invert", action="store_true", help="Force invert")
    ap.add_argument("--no-invert", action="store_true", help="Force no invert")
    ap.add_argument("--threshold", type=int, default=120, help="Threshold for digit mask (try 80/120/160)")
    ap.add_argument("--blur", type=float, default=0.3, help="Gaussian blur radius (0 disables)")
    ap.add_argument("--debug", action="store_true", help="Save debug_28x28.png for each image (overwrites)")
    ap.add_argument("--debug-out", default="debug_28x28.png", help="Debug output filename")
    ap.add_argument("--show-probs", action="store_true", help="Print all class probabilities")
    args = ap.parse_args()

    if (args.image is None) == (args.folder is None):
        raise SystemExit("Provide exactly one of --image or --folder")

    if args.invert and args.no_invert:
        raise SystemExit("Choose only one: --invert or --no-invert")

    invert_flag = None
    if args.invert:
        invert_flag = True
    elif args.no_invert:
        invert_flag = False

    device = get_device()
    ts_model = torch.jit.load(args.model, map_location=device)
    ts_model.eval()

    paths = [args.image] if args.image else list(iter_images(args.folder, args.recursive))
    if not paths:
        raise SystemExit("No images found.")

    for p in paths:
        x = preprocess_mnist_like(
            p,
            invert=invert_flag,
            threshold=args.threshold,
            blur=args.blur,
            debug=args.debug,
            debug_out=args.debug_out,
        )
        pred, conf, probs = predict(ts_model, x, device)
        print(f"{p} -> pred={pred} conf={conf:.4f}")

        if args.show_probs:
            print("  " + " ".join([f"{i}:{probs[i]:.3f}" for i in range(10)]))

    print("\nTip: If accuracy is off, run with --debug and open debug_28x28.png. "
          "Then try --invert or tweak --threshold (80/120/160).")


if __name__ == "__main__":
    main()