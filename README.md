# EDAT

**EDAT** (Efficient Distributed Adversarial Training) — training code for distributed adversarial learning with gradient quantization and PGD / fast (FGSM-style) variants.

This repository builds on ideas from **DAT** ([Distributed Adversarial Training to Robustify Deep Neural Networks at Scale](https://openreview.net/pdf?id=Srgg_ULj9gq)) and reuses components from:

- [You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle (YOPO)](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once)
- [pytorch-lamb](https://github.com/cybertronai/pytorch-lamb) (see `lamb.py`)

Pretrained ImageNet models from the original DAT release are available [on Dropbox](https://www.dropbox.com/sh/bbtyxc8fg8q6sbz/AAB_9FYPhUOvgW7a2yxDN_1Ya?dl=0).

## Requirements

- Python 3, PyTorch (with distributed / NCCL where applicable)
- `torchvision`, `timm`, `tqdm`, `pandas`

Install dependencies as needed for your environment (no `requirements.txt` is bundled in this tree).

## Distributed launch

Training uses `torch.distributed` with `init_method='env://'`. Start jobs with **`torchrun`** (or `python -m torch.distributed.launch`) and set the usual environment variables (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, or use `--standalone` / `--nnodes` as appropriate).

Example — **ImageNet with PGD** (multi-GPU on one node):

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> main_edat.py \
  --dataset imagenet \
  --batch-size <BATCH_SIZE> \
  --num-epochs 30 \
  --lr 0.01
```

Example — **ImageNet with fast FGSM-style training** (`--fast`):

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> main_edat.py \
  --dataset imagenet \
  --batch-size <BATCH_SIZE> \
  --num-epochs 30 \
  --lr 0.01 \
  --fast
```

**Note:** For ImageNet / Tiny-ImageNet, the dataset root is set inside `main_edat.py` (the `root = ...` lines under `elif 'imagenet' in args.dataset`). Point that path at a directory whose layout matches `torchvision.datasets.ImageFolder`: `train/` and `val/` subfolders. `--dataset-path` applies to CIFAR runs, not to the current ImageNet branch. Adjust `CUDA_VISIBLE_DEVICES` near the top of `main_edat.py` if it does not match your machine.

Multi-node training follows the usual PyTorch pattern: set `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and per-process `RANK`, then launch `torchrun` / `torch.distributed.launch` with matching process count (analogous to `--world-size`, `--rank`, and `--dist-url "tcp://<MASTER_IP>:<PORT>"` in classic DAT scripts).

### Other datasets

Supported `--dataset` values include `cifar`, `cifarext`, `imagenet`, and `tinyimagenet`. CIFAR-style runs use `--dataset-path` as the download/root directory for the dataset.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--fast` | Use fast adversarial training branch (FGSM-style) instead of full PGD in the training loop |
| `--wolalr` | Train without layer-wise adaptive LR (uses SGD when enabled; default is on) |
| `--dist-backend` | Distributed backend (default `nccl`) |
| `--output_txt` | Log file name under `outputs/` for eval metrics |

## Project layout (main files)

| File | Role |
|------|------|
| `main_edat.py` | Training entry point |
| `attack.py` | PGD attack |
| `quantization.py` | Gradient quantizers used in `distributed()` |
| `dataset.py` | CIFAR / ImageNet loaders |
| `models.py`, `cifar_resnet18.py`, `wide_resnet.py` | Model definitions |
| `eval.py` | Evaluation utilities |
| `utils.py` | Metrics, checkpoints |

## Citation

If you use this code or the DAT paper, please cite the original DAT work as appropriate:

```bibtex
@inproceedings{dat2022,
  title={Distributed Adversarial Training to Robustify Deep Neural Networks at Scale},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/pdf?id=Srgg_ULj9gq}
}
```

(Add a citation for EDAT if your project publishes one.)
