# BRAVE — Breast video classification (Swin + temporal attention)

*Swin stages plus lightweight temporal Transformers for fixed-length multi-frame video classification.*

Video backbone **BRAVE**: time–batch merge, four hierarchical Swin Transformer stages, and progressive frame-level temporal fusion at every stage, followed by pooled classification.

## Architecture

For the default 16-frame, 224 × 224 input, BRAVE uses the following feature hierarchy:

| Stage | Swin blocks | Patch merging | LTTM | LTTM input |
|------|-------------|---------------|------|------------|
| 1 | 2 blocks, 3 heads | Yes | 1 layer, 2 heads | `[B×16, 28×28, 192]` |
| 2 | 2 blocks, 6 heads | Yes | 1 layer, 3 heads | `[B×16, 14×14, 384]` |
| 3 | 6 blocks, 12 heads | Yes | 2 layers, 6 heads | `[B×16, 7×7, 768]` |
| 4 | 2 blocks, 24 heads | No | 2 layers, 6 heads | `[B×16, 7×7, 768]` |

Each lightweight temporal Transformer module (LTTM) reshapes the stage features
to `[B, 16, C, H×W]`, applies spatial global average pooling, adds a learnable
temporal positional embedding, and processes the resulting frame sequence with
pre-normalized multi-head self-attention and MLP blocks. A linear projection
maps the refined frame features back to the stage channel dimension. These
features are broadcast over the spatial grid and added to the original stage
features through a residual connection.

After Stage 4, layer normalization is followed by temporal summation, adaptive
average pooling over the spatial tokens, and a task-specific fully connected
classification head. The benign-versus-malignant and IBC-versus-DCIS tasks use
separate BRAVE model instances. Their architectures are identical, but they do
not share backbone parameters or classification-head weights.

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
python train.py --data-path /path/to/data --train-xlsx train_f.xlsx --val-xlsx val_f.xlsx --label-col label1 --num-classes 2 --batch-size 2 --epochs 100 --lr 1e-4 --weight-decay 0.05 --data-parallel
```

The default optimizer is AdamW with an initial learning rate of `1e-4`, weight
decay of `0.05`, and cosine annealing over the requested number of epochs
(`--min-lr 0` by default). Here, `--batch-size 2` means two videos per
optimization step. With 16 frames per video, merging the temporal and batch
dimensions produces 32 frame-level inputs to the 2D Swin backbone. It should
not be interpreted as a video batch size of 32.

Training augmentation uses random resized cropping, rotation within ±15°, and
translation by up to 10% of image width and height. RGB means and standard
deviations are calculated from the training videos only and saved as
`normalization.json` beside the model checkpoints. Validation and inference
reuse these saved statistics. Use `--normalization-json` to supply previously
calculated training statistics instead of recalculating them.

Train the two clinical tasks in separate runs with their corresponding patient
lists and label columns. Use `--label-col label1` for the
benign-versus-malignant task and the appropriate malignant-only training list
with `--label-col label2` for the IBC-versus-DCIS task.

Use `--data-parallel` to train on two visible CUDA devices (`cuda:0` and
`cuda:1`). Saved checkpoints are automatically unwrapped and remain compatible
with the single-device prediction script.

Optional: `--weights path/to.pth`, `--freeze-layers`, `--weights-out-dir`,
`--num-frames`, `--min-lr`, `--device cuda:0`.

The four-stage frame-level LTTM changes the temporal-module parameterization.
Legacy BRAVE or 2D Swin checkpoints can still initialize compatible spatial
backbone weights during training, but all LTTMs are reinitialized. When
`--freeze-layers` is used with such a checkpoint, the classification head and
the newly initialized LTTMs remain trainable. Inference requires a checkpoint
trained with the current architecture.

## Evaluate

```bash
python predict.py --excel /path/to/test.xlsx --weights ./weights_new/BRAVE/model-50.pth --num-classes 2
```

Or: `--weights-dir ./weights_new/BRAVE --epoch 50`. Optional: `--out-txt results.txt`, `--label-col label2`.
`predict.py` automatically loads `normalization.json` from the checkpoint
directory, or it can be supplied explicitly with `--normalization-json`.

## Architecture tests

```bash
python -m unittest -v test_model_architecture.py
```

## Citation
