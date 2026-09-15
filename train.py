import os
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet3D
from model import BRAVE
from utils import evaluate, train_one_epoch


def prepare_pretrained_weights(weights_dict):
    weights_dict = dict(weights_dict)
    required_temporal_keys = (
        "temporal_modules.0.temporal_pos_embed",
        "temporal_modules.3.temporal_pos_embed",
        "temporal_modules.3.context_proj.weight",
    )
    reset_temporal = not all(k in weights_dict for k in required_temporal_keys)

    for k in list(weights_dict.keys()):
        if "head" in k or (reset_temporal and k.startswith("temporal_modules.")):
            del weights_dict[k]

    return weights_dict, reset_temporal


def compute_training_channel_stats(excel_file, label_col):
    dataset = MyDataSet3D(
        excel_file=excel_file,
        transform=transforms.ToTensor(),
        label_col=label_col,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for index in range(len(dataset)):
        video, _ = dataset[index]
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(
                "Expected training videos with shape [T, 3, H, W], got {} at index {}".format(
                    tuple(video.shape), index
                )
            )
        video = video.to(torch.float64)
        channel_sum += video.sum(dim=(0, 2, 3))
        channel_sum_sq += video.square().sum(dim=(0, 2, 3))
        pixel_count += video.shape[0] * video.shape[2] * video.shape[3]

    if pixel_count == 0:
        raise ValueError("Cannot compute normalization statistics from an empty training set")

    mean = channel_sum / pixel_count
    variance = channel_sum_sq / pixel_count - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    return mean.tolist(), std.tolist()


def resolve_normalization_stats(args, train_path, label_col):
    if args.normalization_json:
        with open(args.normalization_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mean = payload["mean"]
        std = payload["std"]
    else:
        mean, std = compute_training_channel_stats(train_path, label_col)

    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Normalization mean and standard deviation must contain three RGB values")

    output_path = os.path.join(args.weights_out_dir, "normalization.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": mean,
                "std": std,
                "source": os.path.abspath(train_path),
            },
            f,
            indent=2,
        )
    return mean, std


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.weights_out_dir):
        os.makedirs(args.weights_out_dir)

    tb_writer = SummaryWriter()

    train_path = os.path.join(args.data_path, args.train_xlsx)
    val_path = os.path.join(args.data_path, args.val_xlsx)

    img_size = 224
    normalization_mean, normalization_std = resolve_normalization_stats(
        args, train_path, args.label_col
    )
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomAffine(
                    degrees=args.rotation_degrees,
                    translate=(args.translation_fraction, args.translation_fraction),
                ),
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.143)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(normalization_mean, normalization_std),
            ]
        ),
    }

    train_dataset = MyDataSet3D(
        excel_file=train_path,
        transform=data_transform["train"],
        label_col=args.label_col,
    )
    val_dataset = MyDataSet3D(
        excel_file=val_path,
        transform=data_transform["val"],
        label_col=args.label_col,
    )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    model = BRAVE(num_classes=args.num_classes, num_frames=args.num_frames).to(device)

    reset_temporal = False
    if args.weights:
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict, reset_temporal = prepare_pretrained_weights(weights_dict)
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            train_temporal = reset_temporal and name.startswith("temporal_modules.")
            if "head" not in name and not train_temporal:
                para.requires_grad_(False)

    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
        )
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_acc", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        save_path = os.path.join(args.weights_out_dir, "model-{}.pth".format(epoch))
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), save_path)
        scheduler.step()

    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BRAVE video classifier")
    parser.add_argument(
        "--num-classes",
        "--num_classes",
        dest="num_classes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label2",
        help="Label column for single-task training",
    )
    parser.add_argument(
        "--num-frames",
        "--num_frames",
        dest="num_frames",
        type=int,
        default=16,
        help="Temporal length T; must match dataset frames and BRAVE",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of videos per optimization step; 2 videos × 16 frames = 32 frame inputs",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=15.0,
        help="Maximum absolute random rotation applied during training",
    )
    parser.add_argument(
        "--translation-fraction",
        type=float,
        default=0.10,
        help="Maximum random translation as a fraction of image width and height",
    )
    parser.add_argument(
        "--normalization-json",
        type=str,
        default="",
        help="Optional JSON containing training-set RGB mean and std; otherwise compute them",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine annealing",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/media/ubuntu/8TDisk/乳腺",
    )
    parser.add_argument(
        "--train-xlsx",
        type=str,
        default="train_f.xlsx",
        help="Excel under data-path listing training videos",
    )
    parser.add_argument(
        "--val-xlsx",
        type=str,
        default="val_f.xlsx",
        help="Excel under data-path listing validation videos",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional checkpoint; incompatible temporal modules and head keys are skipped",
    )
    parser.add_argument(
        "--weights-out-dir",
        type=str,
        default="./weights_new/BRAVE",
        help="Directory for saved model-epoch.pth",
    )
    parser.add_argument(
        "--freeze-layers",
        action="store_true",
        help=(
            "Freeze the spatial backbone; train the head and, when loading a "
            "legacy checkpoint, the reinitialized temporal modules"
        ),
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Train on two visible CUDA devices (0 and 1) with DataParallel",
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")

    opt = parser.parse_args()
    main(opt)
