# 减少CNN层数，优化训练速度的修改版本
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import mynn as nn
from draw_tools.plot import plot

# 固定随机种子以便结果可复现
np.random.seed(309)

# 数据加载部分保持不变
train_images_path = r'/home/PJ1/codes/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'/home/PJ1/codes/dataset/MNIST/train-labels-idx1-ubyte.gz'  

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 随机打乱数据
idx = np.random.permutation(np.arange(num))
# 保存索引
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# 划分验证集和测试集
valid_imgs = train_imgs[:800]
valid_labs = train_labs[:800]
test_imgs = train_imgs[800:10000]
test_labs = train_labs[800:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 归一化
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0
test_imgs = test_imgs / 255.0

# 重塑为[batch_size, channels, height, width]
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
test_imgs = test_imgs.reshape(-1, 1, 28, 28)

# 优化CNN架构: 减少层数，并适当减少通道数
# 修改后的卷积层参数 - 从2层减少为1层
optimized_conv_params = [
    {
        'in_channels': 1,
        'out_channels': 16,  # 保持合理的通道数
        'kernel_size': 5,    # 使用更大的核心来捕获更多信息
        'stride': 1,
        'padding': 2         # 保持特征图尺寸
    }
]

# 对应调整全连接层输入尺寸
# 由于只有一层卷积，特征图尺寸保持28x28
fc_size_list = [16 * 28 * 28, 64, 10]  # 减少中间层神经元数量

# 初始化CNN模型
cnn_model = nn.models.Model_CNN(optimized_conv_params, fc_size_list, 'ReLU', [1e-5])

# 优化训练参数: 增大学习率和batch size以加速训练
optimizer = nn.optimizer.SGD(init_lr=0.02, model=cnn_model)  # 增大学习率
# 使用步长更大的学习率衰减
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[400, 800], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=10)

# 增大batch size以加速训练
runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)

# 减少训练轮次
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=3, log_iters=100, save_dir=r'./best_models')

# 在测试集上评估
test_score, test_loss = runner.evaluate([test_imgs, test_labs])
print(f"Test accuracy: {test_score:.4f}, loss: {test_loss:.4f}")

# 可视化训练过程
_, axes = plt.subplots(1, 2)
axes = axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()