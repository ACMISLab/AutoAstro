# 在 'configs/resnet/' 创建此文件
_base_ = './resnet50_8xb32_in1k.py'

# 模型在之前的基础上使用 CutMix 训练增强
model = dict(
    train_cfg=dict(
        augments=dict(type='CutMix', alpha=1.0)
    )
)

# 优化策略在之前基础上训练更多个 epoch
train_cfg = dict(max_epochs=300, val_interval=10)  # 训练 300 个 epoch，每 10 个 epoch 评估一次
param_scheduler = dict(step=[150, 200, 250])   # 学习率调整也有所变动

# 使用自己的数据集目录
train_dataloader = dict(
        dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/train.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='train',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=['A', 'B', 'C', 'D', ...],  # 每个类别的名称
        pipeline=...,    # 处理数据集样本的一系列变换操作
    )
)
val_dataloader = dict(
    batch_size=64,                  # 验证时没有反向传播，可以使用更大的 batchsize
    dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/train.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='train',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=['A', 'B', 'C', 'D', ...],  # 每个类别的名称
        pipeline=...,    # 处理数据集样本的一系列变换操作
    )
)
test_dataloader = dict(
    batch_size=64,                  # 测试时没有反向传播，可以使用更大的 batchsize
    dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/train.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='train',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=['A', 'B', 'C', 'D', ...],  # 每个类别的名称
        pipeline=...,    # 处理数据集样本的一系列变换操作
    )
)