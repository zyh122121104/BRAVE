import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, roc_curve
from torchvision import transforms

from model import BRAVE


def build_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def video_to_batch_tensor(video_np, transform, num_frames):
    if video_np.shape[0] != num_frames:
        raise ValueError(
            "Expected {} frames, got {} (path in Excel)".format(num_frames, video_np.shape[0])
        )
    frames = []
    for frame in video_np:
        frames.append(transform(Image.fromarray(frame)))
    return torch.stack(frames).unsqueeze(0)


def is_sample_row(row, id_col="ID"):
    if id_col not in row or pd.isnull(row[id_col]):
        return False
    s = str(row[id_col]).strip()
    if s in ("", "stop"):
        return False
    return True


def flush_group(x_sum, n, current_label, num_classes, out_fp, out_id, results):
    empty = np.zeros((num_classes,), dtype=np.float64)
    if n == 0 or current_label is None:
        return empty, 0, None
    avg_prob = x_sum / n
    pred_cls = int(np.argmax(avg_prob))
    results["y_true"].append(current_label)
    results["y_pred"].append(pred_cls)
    results["probs"].append(avg_prob.copy())
    if out_fp is not None:
        if num_classes == 2:
            score = float(avg_prob[1])
        else:
            score = " ".join("{:.6f}".format(float(v)) for v in avg_prob)
        line_id = out_id if out_id is not None else ""
        out_fp.write("{}\t{}\t{}\n".format(line_id, score, current_label))
    return empty, 0, None


def resolve_normalization_stats(args, checkpoint_path):
    stats_path = args.normalization_json
    if not stats_path:
        candidate = os.path.join(os.path.dirname(checkpoint_path), "normalization.json")
        if os.path.isfile(candidate):
            stats_path = candidate

    if stats_path:
        with open(stats_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mean = payload["mean"]
        std = payload["std"]
    elif args.norm_mean is not None and args.norm_std is not None:
        mean = args.norm_mean
        std = args.norm_std
    else:
        raise ValueError(
            "Training-set normalization is required. Place normalization.json "
            "beside the checkpoint or provide --normalization-json."
        )

    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Normalization mean and standard deviation must contain three RGB values")
    return mean, std


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.weights_dir is not None and args.epoch is not None:
        ckpt = os.path.join(args.weights_dir, "model-{}.pth".format(args.epoch))
    else:
        ckpt = args.weights
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt))

    normalization_mean, normalization_std = resolve_normalization_stats(args, ckpt)
    transform = build_transform(
        args.img_size, normalization_mean, normalization_std
    )

    model = BRAVE(num_classes=args.num_classes, num_frames=args.num_frames).to(device)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(
            "Checkpoint is incompatible with the four-stage frame-level LTTM architecture. "
            "Use weights trained with the current model.py."
        ) from e
    model.eval()

    data = pd.read_excel(args.excel, dtype=str)
    if args.label_col not in data.columns:
        raise KeyError("Excel has no column {!r}".format(args.label_col))

    out_fp = open(args.out_txt, "w", encoding="utf-8") if args.out_txt else None

    results = {"y_true": [], "y_pred": [], "probs": []}
    x_sum = np.zeros((args.num_classes,), dtype=np.float64)
    n = 0
    current_label = None
    last_id = None

    for _, row in data.iterrows():
        if is_sample_row(row):
            video_path = row["path"]
            video = np.load(video_path)
            tensor = video_to_batch_tensor(video, transform, args.num_frames)
            current_label = int(float(row[args.label_col]))
            last_id = row.get("ID", "")
            with torch.no_grad():
                logits = model(tensor.to(device)).squeeze(0).cpu()
                prob = torch.softmax(logits, dim=0).numpy()
            x_sum += prob
            n += 1
        else:
            x_sum, n, current_label = flush_group(
                x_sum, n, current_label, args.num_classes, out_fp, last_id, results
            )
            last_id = None

    flush_group(x_sum, n, current_label, args.num_classes, out_fp, last_id, results)

    if out_fp is not None:
        out_fp.close()

    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])
    probs = np.stack(results["probs"]) if results["probs"] else np.zeros((0, args.num_classes))

    if len(y_true) == 0:
        print("No completed groups found (check Excel and delimiter rows).")
        return

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("Samples (groups):", len(y_true))
    print("Accuracy:", acc)
    print("F1 (macro):", f1m)

    if args.num_classes == 2:
        pos_scores = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true, pos_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print("AUC (positive class = 1):", roc_auc)
        tp = tn = fp = fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == yt:
                if yt == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if yt == 1:
                    fn += 1
                else:
                    fp += 1
        print("TP FP TN FN:", tp, fp, tn, fn)
        if tp + fn > 0:
            print("Sensitivity (class 1):", tp / (tp + fn))
        if tn + fp > 0:
            print("Specificity (class 0):", tn / (tn + fp))
    else:
        try:
            roc_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            print("AUC (macro OVR):", roc_auc)
        except ValueError as e:
            print("AUC skipped:", e)
        for c in range(args.num_classes):
            mask = y_true == c
            if mask.sum() == 0:
                continue
            sub_acc = (y_pred[mask] == y_true[mask]).mean()
            print("Class {} accuracy: {:.4f}".format(c, sub_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BRAVE on an Excel test list")
    parser.add_argument(
        "--excel",
        type=str,
        required=True,
        help="Path to test Excel (columns: ID, path, label1/label2, ...)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to a single model .pth (use with --weights-dir/--epoch OR this alone)",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Directory containing model-{epoch}.pth",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch index when using --weights-dir",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--normalization-json",
        type=str,
        default="",
        help="Training-set normalization JSON; auto-detected beside the checkpoint",
    )
    parser.add_argument("--norm-mean", type=float, nargs=3, default=None)
    parser.add_argument("--norm-std", type=float, nargs=3, default=None)
    parser.add_argument(
        "--label-col",
        type=str,
        default="label1",
        help="Label column name in Excel (e.g. label1, label2)",
    )
    parser.add_argument(
        "--out-txt",
        type=str,
        default="",
        help="If set, write ID\\tscore(s)\\tlabel per group",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    opt = parser.parse_args()
    if not opt.weights and (opt.weights_dir is None or opt.epoch is None):
        parser.error("Provide either --weights path or both --weights-dir and --epoch")
    main(opt)
