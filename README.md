# CIFAR-10 图像 tokenizer（PyTorch）

这部分实现 MaskGIT 的第一阶段：图像经卷积编码器和 VQ 码本变成离散 token，再经码本和卷积解码器重建图像。训练不需要类别标签，也不依赖 TensorFlow 或 torchvision。

模型沿用原项目的残差块、GroupNorm、SiLU、平均池化下采样、最近邻上采样和最近邻向量量化思路。CIFAR-10 图像只有 32×32，因此默认使用两次下采样，得到 **8×8 个 token**，码本大小为 512。当前训练目标是图像 L1 损失加 VQ 的码本与 commitment 损失；没有加入原 VQGAN 的感知损失和判别器，因此这是可训练的 VQ tokenizer 基线，不等于原论文的完整 VQGAN 训练复现。

`tokenizer` 与 `maskgit` 同级，是可直接运行的脚本工程，不需要作为 Python 包安装或使用 `python -m tokenizer...`。目录结构如下：

```text
tokenizer/
├── train.py          # 训练入口
├── reconstruct.py    # token 编码与图像重建入口
├── model.py          # 编码器、码本和解码器
├── data.py           # CIFAR-10 数据读取
├── image_io.py       # 图像输入输出
├── data/             # 默认数据目录
├── runs/             # 默认训练结果目录
└── outputs/          # 默认重建结果目录（运行时创建）
```

进入 `tokenizer` 目录后运行：

```powershell
cd E:\py_Prjs\maskgit-main\tokenizer
conda activate maskgit-tokenizer
python -m pip install -r requirements.txt
python train.py --epochs 30 --batch-size 128
python reconstruct.py --checkpoint runs/cifar10/best.pt --cifar-index 0
```

训练命令首次运行时会从[数据集官网](https://www.cs.toronto.edu/~kriz/cifar.html)下载 CIFAR-10 二进制包（约 162 MB），核对 MD5 后直接读取压缩包。如果已有官方 `cifar-10-binary.tar.gz` 或 `cifar-10-python.tar.gz`，可以放进 `data/`，或者通过 `--data-dir` 指定所在目录。默认数据与结果目录以 `tokenizer` 为基准，不依赖启动命令时的工作目录。训练集 50,000 张，测试集 10,000 张；标签不会用于训练。

当前机器已经有 `E:\py_Prjs\Datasets\CIFAR-10\cifar-10-python.tar.gz`。在这里运行时，可在训练和重建命令中都加入 `--data-dir "E:\py_Prjs\Datasets\CIFAR-10"`，无需重新下载。

每轮训练在 `runs/cifar10` 保存 `last.pt`、验证损失最好的 `best.pt`，以及 `reconstruction_epoch_XXX.png`。对比图上排是原图，下排是重建图。终端还会输出重建误差、PSNR、码本使用率与困惑度。`--resume runs/cifar10/last.pt --epochs 60` 可续训到第 60 轮。

训练时会显示按 epoch 更新的进度条和当前训练、验证损失；续训时从检查点已完成的 epoch 数开始显示。每轮结束后仍打印完整指标。

重建命令在 `outputs/` 保存：

- `tokens.npy`：8×8 的整数 token ID，范围为 0～511。
- `original.png`、`reconstruction.png`：原图和重建图。
- `comparison.png`：上下排列的对比图。

也可以传入任意图片：

```powershell
python reconstruct.py --checkpoint runs/cifar10/best.pt --image path/to/image.png
```

自定义图片会被缩放至 32×32。小规模检查可在训练命令后加 `--epochs 1 --max-train-batches 2 --max-eval-batches 1`；这样的权重只用于验证程序，不能用于评价重建质量。

待 tokenizer 训练稳定后，可在该目录通过 `from model import ImageTokenizer` 使用 `ImageTokenizer.encode_to_indices(images)` 批量生成离散 token，冻结其权重，再训练后续的 MaskGIT 掩码预测模型。
