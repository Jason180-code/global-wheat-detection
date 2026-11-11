
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--image-list", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
args = parser.parse_args()

images = [Path(p) for p in json.loads(Path(args.image_list).read_text())]
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"
use_half = torch.cuda.is_available()

print(
    f"fold{args.fold}: device={device}, "
    f"images={len(images)}, half={use_half}",
    flush=True,
)

model = YOLO(args.weights)
for index, image_path in enumerate(images, 1):
    result = model.predict(
        source=str(image_path),
        imgsz=int(os.getenv("GW_IMGSZ", "1024")),
        conf=float(os.getenv("GW_CONF", "0.10")),
        iou=float(os.getenv("GW_IOU", "0.70")),
        max_det=int(os.getenv("GW_MAX_DET", "300")),
        device=device,
        augment=os.getenv("GW_TTA", "0") == "1",
        half=use_half,
        batch=1,
        verbose=False,
        workers=0,
    )[0]

    boxes = result.boxes
    target = out_dir / f"{image_path.stem}.npz"

    if boxes is None or len(boxes) == 0:
        np.savez_compressed(
            target,
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )
    else:
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)

        height, width = result.orig_shape[:2]

        xyxy[:, [0, 2]] /= float(width)
        xyxy[:, [1, 3]] /= float(height)
        xyxy = np.clip(xyxy, 0.0, 1.0)

        keep = (
            (xyxy[:, 2] > xyxy[:, 0])
            & (xyxy[:, 3] > xyxy[:, 1])
        )
        xyxy = xyxy[keep]
        scores = scores[keep]

        order = np.argsort(-scores)[:300]

        np.savez_compressed(
            target,
            boxes=xyxy[order],
            scores=scores[order],
        )

    del result

    if index % 100 == 0 or index == len(images):
        print(
            f"fold{args.fold}: {index}/{len(images)}",
            flush=True,
        )

print(f"fold{args.fold}: finished", flush=True)
