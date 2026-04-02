# EDAT

**Efficient Distributed Adversarial Training（高效分布式对抗训练）**

本仓库实现基于 PyTorch 分布式训练的对抗鲁棒性学习流程：在多台 GPU 上使用 **PGD** 生成对抗样本进行训练，并通过 **梯度量化 / 压缩**（如随机量化）降低多机通信开销。代码中还包含 **Lamb** 优化器、多种骨干网络与数据集加载逻辑。

## 功能概览

- **分布式训练**：`torch.distributed` + `DistributedDataParallel`，默认后端 `nccl`，通过环境变量初始化（`env://`）。
- **对抗攻击**：`attack.py` 中的 **PGD**（投影梯度下降），支持自定义 \(\epsilon\)、步长与迭代次数；可选快速 **FGSM** 风格路径（`--fast`）。
- **通信压缩**：`quantization.py` 提供 `RandomQuantizer`、`NCQuantizer`、`KContractionQuantizer` 等，在 `main_edat.py` 中通过 `distributed()` 对梯度做聚合前的量化与反量化（可在源码中切换量化器）。
- **数据集**：CIFAR-100（`cifar`）、CIFAR-10 与伪标签扩充数据（`cifarext`，需 `ti_500K_pseudo_labeled.pickle`）、ImageNet / Tiny-ImageNet 风格的 `ImageFolder` 加载（`imagenet` / `tinyimagenet`）。
- **模型**：CIFAR 路径当前使用 `torchvision` 的 **ResNet152**（预训练）；ImageNet 路径使用 **timm** 的 `deit_tiny_patch16_224`（预训练）。`models.py` 另含 PreAct-ResNet、CIFAR ResNet 等定义，可按需替换 `main_edat.py` 中的网络构建。
- **独立评估**：`eval.py` 用于在指定 checkpoint 与数据集上评估干净准确率与 PGD 鲁棒准确率。

## 环境依赖

- Python 3
- [PyTorch](https://pytorch.org/)（需 CUDA，多卡训练）
- `torchvision`
- `tqdm`
- `timm`（ImageNet / DeiT 路径）
- `pandas`（`main_edat.py` 已导入）

安装示例（请按本机 CUDA 版本从 PyTorch 官网选择对应命令）：

```bash
pip install torch torchvision tqdm timm pandas
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `main_edat.py` | 分布式对抗训练主入口：训练循环、梯度聚合与量化、`eval` 与日志 |
| `attack.py` | PGD 等攻击实现（部分思路参考 YOPO 相关代码注释） |
| `quantization.py` | 梯度量化器（随机量化、自然压缩、Top-K 等） |
| `dataset.py` | CIFAR / ImageNet 数据加载与 `DistributedSampler` |
| `models.py` | PreAct-ResNet、ResNet 变体等 |
| `cifar_resnet18.py` / `wide_resnet.py` | 其他 CIFAR 常用骨干（若被引用） |
| `lamb.py` | Lamb 优化器（来自 pytorch-lamb 思路） |
| `utils.py` | 准确率、`AvgMeter`、checkpoint 等工具 |
| `eval.py` | 单独评估脚本 |

## 数据准备

- **`--dataset cifar`**：在 `--dataset-path`（默认 `datasets`）下由 `torchvision` 自动下载 **CIFAR-100**。
- **`--dataset cifarext`**：需要 **CIFAR-10** 数据及同目录下的 `ti_500K_pseudo_labeled.pickle`（脚本会打印加载信息）。
- **`--dataset imagenet` / `tinyimagenet`**：`dataset.py` 使用 `ImageFolder`，需 `root/train` 与 `root/val` 子目录。注意 **`main_edat.py` 中对 ImageNet / Tiny-ImageNet 的根路径有硬编码**，使用前请改为本机路径或通过修改源码与 `--dataset-path` 保持一致。

训练日志默认追加写入 `outputs/` 下由 `--output_txt` 指定的文件；若目录不存在需自行创建或先在代码中创建。

## 运行方式

本脚本通过 **`dist.init_process_group(..., init_method='env://')`** 启动，需用 PyTorch 分布式启动器并设置进程数与世界规模，例如单机多卡：

```bash
# 4 张 GPU 示例（需与代码中 CUDA 设备配置一致）
set CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 main_edat.py --dataset cifar --dataset-path datasets --batch-size 512 --num-epochs 100
```

Linux / macOS 可将 `set` 换为 `export`。`main_edat.py` 顶部含有 `os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'`，请按实际机器修改或删除以避免冲突。

常用参数：

| 参数 | 含义 |
|------|------|
| `--dataset` | `cifar` \| `cifarext` \| `imagenet` \| `tinyimagenet` |
| `--dataset-path` | 数据集根目录（CIFAR 下载目录；ImageNet 见上文硬编码说明） |
| `--batch-size` | 每卡 batch 大小 |
| `--num-epochs` | 训练轮数 |
| `--lr` | 学习率（默认 0.01） |
| `--fast` | 使用快速 FGSM 风格训练分支 |
| `--wolalr` | 为 True 时使用 SGD；否则使用 Lamb（与层自适应学习率相关） |
| `--dist-backend` | 分布式后端，默认 `nccl` |
| `--output_txt` | 输出日志文件名（位于 `outputs/`） |

量化器在 `main_edat.py` 中通过注释切换，例如：

```python
qt = RandomQuantizer()
# qt = NCQuantizer()
# qt = KContractionQuantizer()
```

单独评估可参考 `eval.py` 的参数（`--dataset`、`--dataset-path`、`--checkpoint` 等）。

## 许可证

本项目以 [MIT License](LICENSE) 发布，Copyright (c) 2024 Xjchen。

## 引用与致谢

- PGD 等实现注释中提及 [YOPO-You-Only-Propagate-Once](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once)。
- `utils.py` 中准确率等工具同样标注来源于上述仓库。
- `lamb.py` 思路来自 [pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)。

若本工作对应论文发表信息，请在引用时补充论文条目与正式名称 **EDAT**。
