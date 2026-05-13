# BRAVE — Breast video classification (Swin + temporal attention)

*Swin stages plus lightweight temporal Transformers for fixed-length multi-frame video classification.*

Video backbone **BRAVE**: time–batch merge, Swin Transformer stages with lightweight temporal modules after each patch merge, then pooled classification.

## Environment

Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

Install **PyTorch** with the correct CUDA/CPU wheel from [pytorch.org](https://pytorch.org/get-started/locally/) if the default `pip install -r requirements.txt` does not match your GPU.

## Project layout

| File | Role |
|------|------|
| `model.py` | BRAVE + Swin building blocks |
| `train.py` | Train BRAVE on `MyDataSet3D` |
| `predict.py` | Evaluate on Excel test lists |
| `my_dataset.py` | `MyDataSet3D` / `MyDataSet2D` loaders |
| `utils.py` | `train_one_epoch`, `evaluate`, helpers |

## Data

- Training/validation: Excel columns include `path` (`.npy` video array) and labels; 3D dataset stacks frames to `[B, T, C, H, W]` with **T = 16** by default (must match `BRAVE(..., num_frames=16)`).
- Test Excel for `predict.py`: grouped rows with `ID`, `path`, and label column (`label1` / `label2`); delimiter rows end each case (see `predict.py` docstring).

## Train

```bash
python train.py --data-path /path/to/data --train-xlsx train_f.xlsx --val-xlsx val_f.xlsx --num-classes 2 --batch-size 2 --epochs 100
```

Optional: `--weights path/to.pth`, `--freeze-layers`, `--weights-out-dir`, `--num-frames`, `--device cuda:0`.

## Evaluate

```bash
python predict.py --excel /path/to/test.xlsx --weights ./weights_new/BRAVE/model-50.pth --num-classes 2
```

Or: `--weights-dir ./weights_new/BRAVE --epoch 50`. Optional: `--out-txt results.txt`, `--label-col label2`.

## Citation
