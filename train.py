import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet3D
from model import BRAVE
from utils import train_one_epoch, evaluate


def main(args):
    """Train BRAVE on 3D video clips (MyDataSet3D, shape [B, T, C, H, W])."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    if not os.path.exists(args.weights_out_dir):
        os.makedirs(args.weights_out_dir)

    tb_writer = SummaryWriter()

    train_path = os.path.join(args.data_path, args.train_xlsx)
    val_path = os.path.join(args.data_path, args.val_xlsx)

    img_size = 224
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.143)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    }

    train_dataset = MyDataSet3D(excel_file=train_path, transform=data_transform["train"])
    val_dataset = MyDataSet3D(excel_file=val_path, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers".format(nw))

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

    if args.weights:
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)

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

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        save_path = os.path.join(args.weights_out_dir, "model-{}.pth".format(epoch))
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BRAVE video classifier")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Temporal length T; must match dataset frames and BRAVE",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
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
        help="Optional checkpoint; head keys are skipped for partial load",
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
        help="Freeze all parameters except the classification head",
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")

    opt = parser.parse_args()
    main(opt)
