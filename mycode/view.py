import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
from struct import unpack
import os
from tqdm import tqdm
import sys
from matplotlib.gridspec import GridSpec

# 确保当前目录在系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 定义所有需要的类，确保pickle可以在加载时找到这些类
class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_dim, out_dim, weight_scale=1e-2):
        super().__init__()
        self.params['W'] = np.random.randn(in_dim, out_dim) * weight_scale
        self.params['b'] = np.zeros(out_dim)
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params['W'] + self.params['b']
    
    def backward(self, grad):
        self.grads['W'] = self.inputs.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        return grad @ self.params['W'].T

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None
        
    def forward(self, inputs):
        self.mask = inputs > 0
        return inputs * self.mask
    
    def backward(self, grad):
        return grad * self.mask

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        weight_scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.params['W'] = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) * weight_scale
        self.params['b'] = np.zeros(out_channels)
        
        self.inputs = None
        self.padded = None
        self.output_shape = None
    
    def im2col(self, x, h_out, w_out):
        N, C, H, W = x.shape
        k = self.kernel_size
        
        cols = np.zeros((N, C, k, k, h_out, w_out))
        
        for h in range(h_out):
            h_start = h * self.stride
            for w in range(w_out):
                w_start = w * self.stride
                cols[:, :, :, :, h, w] = x[:, :, h_start:h_start+k, w_start:w_start+k]
        
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * h_out * w_out, -1)
        return cols
    
    def forward(self, inputs):
        N, C, H, W = inputs.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        if p > 0:
            self.padded = np.pad(inputs, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        else:
            self.padded = inputs
            
        h_out = (self.padded.shape[2] - k) // s + 1
        w_out = (self.padded.shape[3] - k) // s + 1
        self.output_shape = (N, self.out_channels, h_out, w_out)
        
        W_col = self.params['W'].reshape(self.out_channels, -1)
        
        self.inputs = inputs
        x_col = self.im2col(self.padded, h_out, w_out)
        
        output = (W_col @ x_col.T).T
        output = output.reshape(N, h_out, w_out, self.out_channels).transpose(0, 3, 1, 2)
        
        output += self.params['b'].reshape(1, self.out_channels, 1, 1)
        
        return output
    
    def col2im(self, cols, x_shape):
        N, C, H, W = x_shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        H_padded, W_padded = H + 2 * p, W + 2 * p
        h_out = (H_padded - k) // s + 1
        w_out = (W_padded - k) // s + 1
        
        x_padded = np.zeros((N, C, H_padded, W_padded))
        
        cols_reshaped = cols.reshape(N, h_out, w_out, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        
        for h in range(h_out):
            h_start = h * s
            for w in range(w_out):
                w_start = w * s
                x_padded[:, :, h_start:h_start+k, w_start:w_start+k] += cols_reshaped[:, :, :, :, h, w]
        
        if p > 0:
            return x_padded[:, :, p:-p, p:-p]
        return x_padded
    
    def backward(self, grad):
        N, C, H, W = self.inputs.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        h_out = (H + 2 * p - k) // s + 1
        w_out = (W + 2 * p - k) // s + 1
        
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        x_col = self.im2col(self.padded, h_out, w_out)
        
        self.grads['W'] = (grad_reshaped.T @ x_col).reshape(self.params['W'].shape)
        self.grads['b'] = np.sum(grad_reshaped, axis=0)
        
        W_col = self.params['W'].reshape(self.out_channels, -1)
        dx_col = grad_reshaped @ W_col
        
        dx_padded = self.col2im(dx_col.reshape(N, h_out, w_out, -1), (N, C, H, W))
        
        if p > 0:
            dx = dx_padded[:, :, p:-p, p:-p]
        else:
            dx = dx_padded
            
        return dx

class MaxPool2D(Layer):
    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = pool_size if stride is None else stride
        self.inputs = None
        self.mask = None
    
    def forward(self, inputs):
        N, C, H, W = inputs.shape
        pool = self.pool_size
        stride = self.stride
        
        h_out = (H - pool) // stride + 1
        w_out = (W - pool) // stride + 1
        
        output = np.zeros((N, C, h_out, w_out))
        self.mask = np.zeros_like(inputs)
        
        for n in range(N):
            for c in range(C):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * stride
                        w_start = w * stride
                        
                        patch = inputs[n, c, h_start:h_start+pool, w_start:w_start+pool]
                        
                        max_val = np.max(patch)
                        max_idx = np.argmax(patch)
                        
                        max_h, max_w = np.unravel_index(max_idx, (pool, pool))
                        
                        output[n, c, h, w] = max_val
                        
                        self.mask[n, c, h_start + max_h, w_start + max_w] = 1
        
        self.inputs = inputs
        return output
    
    def backward(self, grad):
        N, C, H_out, W_out = grad.shape
        dx = np.zeros_like(self.inputs)
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        mask_patch = self.mask[n, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                        dx[n, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] += mask_patch * grad[n, c, h, w]
        
        return dx

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)
    
    def backward(self, grad):
        return grad.reshape(self.input_shape)

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, inputs, train=True):
        if train:
            self.mask = np.random.binomial(1, 1-self.p, inputs.shape) / (1-self.p)
            return inputs * self.mask
        return inputs
    
    def backward(self, grad):
        return grad * self.mask

class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.params['gamma'] = np.ones(num_features)
        self.params['beta'] = np.zeros(num_features)
        
        # 运行时统计量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # 缓存
        self.input_shape = None
        self.batch_mean = None
        self.batch_var = None
        self.x_norm = None
        self.inputs = None
    
    def forward(self, inputs, train=True):
        self.inputs = inputs
        self.input_shape = inputs.shape
        
        # 处理不同的输入维度
        if len(inputs.shape) == 4:  # (N, C, H, W) for CNN
            N, C, H, W = inputs.shape
            x_reshaped = inputs.transpose(0, 2, 3, 1).reshape(-1, C)
        else:  # (N, D) for MLP
            x_reshaped = inputs
        
        if train:
            # 计算批次统计量
            self.batch_mean = np.mean(x_reshaped, axis=0)
            self.batch_var = np.var(x_reshaped, axis=0)
            
            # 更新运行统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # 归一化
            self.x_norm = (x_reshaped - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            out = self.params['gamma'] * self.x_norm + self.params['beta']
        else:
            # 使用运行统计量
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.params['gamma'] * x_norm + self.params['beta']
        
        # 恢复原始形状
        if len(self.input_shape) == 4:
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return out
    
    def backward(self, grad):
        if len(self.input_shape) == 4:  # (N, C, H, W)
            N, C, H, W = self.input_shape
            grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, C)
            x_reshaped = self.inputs.transpose(0, 2, 3, 1).reshape(-1, C)
        else:  # (N, D)
            grad_reshaped = grad
            x_reshaped = self.inputs
            
        m = x_reshaped.shape[0]
        
        # 计算梯度
        dx_norm = grad_reshaped * self.params['gamma']
        
        # 反向传播通过归一化
        dvar = np.sum(dx_norm * (x_reshaped - self.batch_mean) * (-0.5) * 
                      np.power(self.batch_var + self.eps, -1.5), axis=0)
        dmean = np.sum(dx_norm * (-1) / np.sqrt(self.batch_var + self.eps), axis=0) + \
                dvar * np.sum(-2 * (x_reshaped - self.batch_mean), axis=0) / m
        
        dx = dx_norm / np.sqrt(self.batch_var + self.eps) + \
             dvar * 2 * (x_reshaped - self.batch_mean) / m + dmean / m
        
        # 可学习参数的梯度
        self.grads['gamma'] = np.sum(grad_reshaped * self.x_norm, axis=0)
        self.grads['beta'] = np.sum(grad_reshaped, axis=0)
        
        # 恢复原始形状
        if len(self.input_shape) == 4:
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return dx

class SimpleResidualBlock(Layer):
    def __init__(self, in_channels, out_channels, use_bn=False, apply_pool=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.apply_pool = apply_pool
        
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        
        self.cache = {}
        
        if use_bn:
            self.bn1 = BatchNorm(out_channels)
            self.bn2 = BatchNorm(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if use_bn:
                self.shortcut_bn = BatchNorm(out_channels)
        else:
            self.shortcut = None

        if apply_pool:
            self.pool = MaxPool2D(2)
        else:
            self.pool = None

    def forward(self, x, train=True):
        self.cache['input'] = x.copy()
        
        identity = x
        
        out = self.conv1.forward(x)
        self.cache['conv1_out'] = out.copy()
        
        if self.use_bn:
            out = self.bn1.forward(out, train)
            self.cache['bn1_out'] = out.copy()
            
        out = self.relu1.forward(out)
        self.cache['relu1_out'] = out.copy()

        out = self.conv2.forward(out)
        self.cache['conv2_out'] = out.copy()
        
        if self.use_bn:
            out = self.bn2.forward(out, train)
            self.cache['bn2_out'] = out.copy()

        if self.shortcut is not None:
            identity = self.shortcut.forward(x)
            self.cache['shortcut_out'] = identity.copy()
            
            if self.use_bn:
                identity = self.shortcut_bn.forward(identity, train)
                self.cache['shortcut_bn_out'] = identity.copy()
        
        out = out + identity
        self.cache['pre_relu2'] = out.copy()
        
        out = self.relu2.forward(out)
        self.cache['relu2_out'] = out.copy()

        if self.pool:
            out = self.pool.forward(out)
            self.cache['pool_out'] = out.copy()

        return out

    def backward(self, grad):
        if self.pool:
            grad = self.pool.backward(grad)
        
        grad = self.relu2.backward(grad)
        
        didentity = grad.copy()
        dmain = grad.copy()
        
        if self.use_bn:
            dmain = self.bn2.backward(dmain)
        
        dmain = self.conv2.backward(dmain)
        dmain = self.relu1.backward(dmain)
        
        if self.use_bn:
            dmain = self.bn1.backward(dmain)
        
        dmain = self.conv1.backward(dmain)

        if self.shortcut is not None:
            if self.use_bn:
                didentity = self.shortcut_bn.backward(didentity)
            didentity = self.shortcut.backward(didentity)
        
        return dmain + didentity

class Softmax:
    def __init__(self):
        self.probs = None
    
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.probs

class CrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()
        self.labels = None
        self.probs = None
    
    def forward(self, inputs, labels):
        self.labels = labels
        self.probs = self.softmax.forward(inputs)
        
        N = inputs.shape[0]
        loss = -np.sum(np.log(self.probs[np.arange(N), labels] + 1e-10)) / N
        return loss
    
    def backward(self):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        grad /= N
        return grad

class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.dropout_layers = []
        self.t = 0  # 用于Adam
        
    def add(self, layer):
        self.layers.append(layer)
        if isinstance(layer, Dropout):
            self.dropout_layers.append(layer)
    
    def set_loss(self, loss_function):
        self.loss_function = loss_function
    
    def forward(self, X, train=True):
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNorm) or isinstance(layer, SimpleResidualBlock):
                outputs = layer.forward(outputs, train)
            else:
                outputs = layer.forward(outputs)
        return outputs
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self, lr, optimizer='sgd', beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.9, weight_decay=1e-4):
        # 初始化Adam的时间步
        if optimizer == 'adam':
            self.t += 1
        
        # 更新每层的参数
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for param in layer.params:
                    # 应用L2正则化
                    if param == 'W':  # 仅对权重应用权重衰减，不对偏置
                        grad = layer.grads[param] + weight_decay * layer.params[param]
                    else:
                        grad = layer.grads[param]
                    
                    if optimizer == 'sgd':
                        # SGD with momentum
                        # 初始化动量缓冲区（如果不存在）
                        if f'v_{param}' not in layer.__dict__:
                            layer.__dict__[f'v_{param}'] = np.zeros_like(layer.params[param])
                        
                        # 更新动量
                        layer.__dict__[f'v_{param}'] = momentum * layer.__dict__[f'v_{param}'] - lr * grad
                        
                        # 更新参数
                        layer.params[param] += layer.__dict__[f'v_{param}']
                    
                    elif optimizer == 'adam':
                        # 初始化动量和平方梯度缓冲区
                        if f'm_{param}' not in layer.__dict__:
                            layer.__dict__[f'm_{param}'] = np.zeros_like(layer.params[param])
                            layer.__dict__[f'v_{param}'] = np.zeros_like(layer.params[param])
                        
                        # 更新有偏一阶矩估计
                        layer.__dict__[f'm_{param}'] = beta1 * layer.__dict__[f'm_{param}'] + (1 - beta1) * grad
                        
                        # 更新有偏二阶矩估计
                        layer.__dict__[f'v_{param}'] = beta2 * layer.__dict__[f'v_{param}'] + (1 - beta2) * (grad ** 2)
                        
                        # 偏差修正
                        m_corrected = layer.__dict__[f'm_{param}'] / (1 - beta1 ** self.t)
                        v_corrected = layer.__dict__[f'v_{param}'] / (1 - beta2 ** self.t)
                        
                        # 更新参数
                        layer.params[param] -= lr * m_corrected / (np.sqrt(v_corrected) + eps)
    
    def predict(self, X):
        outputs = self.forward(X, train=False)
        return np.argmax(outputs, axis=1)
    
    def evaluate(self, X, y):
        """在给定数据和标签上评估模型"""
        # 检查X和y具有相同数量的样本
        assert X.shape[0] == y.shape[0], f"X和y必须具有相同数量的样本，得到 {X.shape[0]} 和 {y.shape[0]}"
        
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

def load_mnist():
    """加载MNIST数据集"""
    try:
        # 尝试加载测试数据
        with gzip.open('/home/PJ1/codes/dataset/MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        
        with gzip.open('/home/PJ1/codes/dataset/MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            y_test = np.frombuffer(f.read(), dtype=np.uint8)
    except FileNotFoundError:
        # 如果当前目录找不到，尝试其他可能的路径
        try:
            possible_paths = [
                '.',
                './data',
                './dataset',
                './MNIST',
                '../data',
                '../dataset',
                '../MNIST',
            ]
            
            for base_path in possible_paths:
                test_images_path = os.path.join(base_path, 't10k-images-idx3-ubyte.gz')
                test_labels_path = os.path.join(base_path, 't10k-labels-idx1-ubyte.gz')
                
                if os.path.exists(test_images_path) and os.path.exists(test_labels_path):
                    with gzip.open(test_images_path, 'rb') as f:
                        magic, num, rows, cols = unpack('>4I', f.read(16))
                        X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
                    
                    with gzip.open(test_labels_path, 'rb') as f:
                        magic, num = unpack('>2I', f.read(8))
                        y_test = np.frombuffer(f.read(), dtype=np.uint8)
                    
                    print(f"在路径 {base_path} 找到MNIST测试数据")
                    break
            else:
                print("未能找到MNIST测试数据")
                return None, None
        except Exception as e:
            print(f"加载数据出错: {e}")
            return None, None
            
    # 归一化数据
    X_test = X_test.astype(np.float32) / 255.0
    
    print(f"成功加载MNIST测试数据: {X_test.shape}")
    return X_test, y_test

def load_model(model_path):
    """加载已保存的模型"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"成功加载模型: {model_path}")
        
        # 打印模型结构
        print("\n模型结构:")
        for i, layer in enumerate(model.layers):
            layer_info = f"第{i+1}层: {layer.__class__.__name__}"
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    layer_info += f", {param_name} 形状: {param.shape}"
            print(layer_info)
        
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def find_model_files(models_dir="./models"):
    """查找所有模型文件"""
    if not os.path.exists(models_dir):
        print(f"目录 {models_dir} 不存在")
        return []
    
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                  if f.endswith('.pkl') and os.path.isfile(os.path.join(models_dir, f))]
    
    if not model_files:
        print(f"在 {models_dir} 目录中没有找到.pkl模型文件")
    else:
        print(f"找到 {len(model_files)} 个模型文件:")
        for i, path in enumerate(model_files, 1):
            print(f"  {i}. {os.path.basename(path)}")
    
    return model_files

def visualize_all_layers(model, X_test, sample_idx=0, save_dir='layer_visualizations'):
    """可视化模型的每一层"""
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 确保输入是CNN格式 (N, C, H, W)
    if len(X_test.shape) == 2:  # 如果是 (N, 784)
        X_cnn = X_test.reshape(-1, 1, 28, 28)
    else:
        X_cnn = X_test
    
    # 获取单个样本
    single_sample = X_cnn[sample_idx:sample_idx+1]
    
    # 存储每层的输出
    layer_outputs = []
    layer_names = []
    
    # 首先添加输入层
    layer_outputs.append(single_sample)
    layer_names.append("Input")
    
    # 前向传播，收集每层输出
    output = single_sample
    for i, layer in enumerate(model.layers):
        layer_name = f"{i+1}: {layer.__class__.__name__}"
        
        # 根据层类型调用适当的forward
        if isinstance(layer, Dropout) or isinstance(layer, BatchNorm) or isinstance(layer, SimpleResidualBlock):
            output = layer.forward(output, train=False)
        else:
            output = layer.forward(output)
        
        layer_outputs.append(output)
        layer_names.append(layer_name)
        
        print(f"处理层 {layer_name}, 输出形状: {output.shape}")
    
    # 获取预测
    prediction = np.argmax(layer_outputs[-1])
    print(f"\n样本 {sample_idx} 预测为: {prediction}")
    
    # 可视化每一层
    for i, (output, name) in enumerate(zip(layer_outputs, layer_names)):
        plt.figure(figsize=(12, 10))
        
        #
        if "Input" in name:
            # 如果是输入层，显示原始图像
            plt.suptitle(f"输入图像 (样本 {sample_idx})", fontsize=16)
            plt.imshow(output[0, 0], cmap='gray')
            plt.axis('off')
        
        elif len(output.shape) == 4 and "Conv" in name:
            # 卷积层输出的特征图可视化
            plt.suptitle(f"层 {name} 特征图 (预测: {prediction})", fontsize=16)
            
            # 获取通道数
            n_channels = output.shape[1]
            
            # 最多显示64个通道，按8x8排列
            n_display = min(64, n_channels)
            n_rows = int(np.ceil(np.sqrt(n_display)))
            n_cols = int(np.ceil(n_display / n_rows))
            
            # 显示每个通道的特征图
            for c in range(n_display):
                plt.subplot(n_rows, n_cols, c + 1)
                
                # 获取特征图
                feature_map = output[0, c]
                
                # 归一化特征图以便可视化
                if np.max(feature_map) != np.min(feature_map):
                    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
                
                plt.imshow(feature_map, cmap='viridis')
                plt.title(f'Channel {c+1}')
                plt.axis('off')
        
        elif len(output.shape) == 4 and "Pool" in name:
            # 池化层输出的特征图可视化
            plt.suptitle(f"层 {name} 特征图 (预测: {prediction})", fontsize=16)
            
            # 获取通道数
            n_channels = output.shape[1]
            
            # 最多显示64个通道，按8x8排列
            n_display = min(64, n_channels)
            n_rows = int(np.ceil(np.sqrt(n_display)))
            n_cols = int(np.ceil(n_display / n_rows))
            
            # 显示每个通道的特征图
            for c in range(n_display):
                plt.subplot(n_rows, n_cols, c + 1)
                
                # 获取特征图
                feature_map = output[0, c]
                
                # 归一化特征图以便可视化
                if np.max(feature_map) != np.min(feature_map):
                    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
                
                plt.imshow(feature_map, cmap='viridis')
                plt.title(f'Channel {c+1}')
                plt.axis('off')
        
        elif len(output.shape) == 2:
            # 全连接层或展平层的输出可视化
            plt.suptitle(f"层 {name} 激活值 (预测: {prediction})", fontsize=16)
            
            # 如果是最终输出层（有10个输出对应数字0-9）
            if output.shape[1] == 10 and i == len(layer_outputs) - 1:
                bars = plt.bar(range(10), output[0])
                plt.xlabel('数字类别')
                plt.ylabel('激活值')
                plt.title('输出层激活值')
                plt.xticks(range(10))
                
                # 高亮预测的类别
                bars[prediction].set_color('red')
            
            # 如果神经元太多，就显示分布而不是具体值
            elif output.shape[1] > 100:
                plt.hist(output[0], bins=50)
                plt.xlabel('激活值')
                plt.ylabel('频率')
                plt.title(f'{name} 激活值分布')
                plt.grid(True, alpha=0.3)
            
            # 否则显示每个神经元的激活值
            else:
                plt.bar(range(output.shape[1]), output[0])
                plt.xlabel('神经元索引')
                plt.ylabel('激活值')
                plt.title(f'{name} 神经元激活值')
        
        else:
            # 其他类型的层，显示形状信息
            plt.suptitle(f"层 {name} (形状: {output.shape})")
            plt.text(0.5, 0.5, f"层类型: {name}\n输出形状: {output.shape}",
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=14,
                     transform=plt.gca().transAxes)
            plt.axis('off')
        
        # 保存图像
        save_path = os.path.join(save_dir, f'layer_{i:02d}_{name.replace(":", "_")}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"已保存层 {name} 的可视化到: {save_path}")
    
    print(f"\n所有层的可视化已保存到目录: {save_dir}")
    return layer_outputs, layer_names

def visualize_weight_filters(model, save_dir='weight_visualizations'):
    """可视化模型的卷积核权重"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 找出所有的卷积层
    conv_layers = [(i, layer) for i, layer in enumerate(model.layers) 
                  if isinstance(layer, Conv2D)]
    
    if not conv_layers:
        print("模型中没有找到卷积层")
        return
    
    # 可视化每个卷积层的权重
    for layer_idx, layer in conv_layers:
        weights = layer.params['W']
        out_channels, in_channels, k_height, k_width = weights.shape
        
        print(f"可视化第{layer_idx+1}层卷积权重: 形状 {weights.shape}")
        
        # 创建图形
        plt.figure(figsize=(15, out_channels // 4 + 5))
        plt.suptitle(f'第{layer_idx+1}层卷积滤波器', fontsize=16)
        
        # 确定每个滤波器的布局
        n_rows = int(np.ceil(out_channels / 8))
        n_cols = min(8, out_channels)
        
        # 可视化每个输出通道的滤波器
        for j in range(out_channels):
            plt.subplot(n_rows, n_cols, j + 1)
            
            # 如果有多个输入通道，计算平均值
            if in_channels > 1:
                filter_img = np.mean(weights[j], axis=0)
                plt.title(f'Filter {j+1}\n(avg of {in_channels} channels)')
            else:
                filter_img = weights[j, 0]
                plt.title(f'Filter {j+1}')
            
            # 归一化以便可视化
            v_min, v_max = filter_img.min(), filter_img.max()
            if v_min != v_max:
                filter_img = (filter_img - v_min) / (v_max - v_min)
            
            plt.imshow(filter_img, cmap='viridis')
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 为suptitle留出空间
        
        # 保存图像
        save_path = os.path.join(save_dir, f'conv_filters_layer_{layer_idx+1}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"已保存第{layer_idx+1}层卷积滤波器可视化到: {save_path}")

def visualize_fc_weights(model, save_dir='weight_visualizations'):
    """可视化全连接层的权重"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 找出所有的全连接层
    fc_layers = [(i, layer) for i, layer in enumerate(model.layers) 
                if isinstance(layer, Linear)]
    
    if not fc_layers:
        print("模型中没有找到全连接层")
        return
    
    # 对每个全连接层可视化权重
    for layer_idx, layer in fc_layers:
        weights = layer.params['W']
        in_features, out_features = weights.shape
        
        print(f"可视化第{layer_idx+1}层全连接层权重: 形状 {weights.shape}")
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        plt.title(f'第{layer_idx+1}层全连接层权重热力图')
        
        # 如果权重太大，只显示一部分
        if in_features > 100 or out_features > 100:
            # 取前100个输入和输出特征
            display_weights = weights[:min(100, in_features), :min(100, out_features)]
            plt.imshow(display_weights, cmap='coolwarm', aspect='auto')
            plt.title(f'第{layer_idx+1}层全连接层权重热力图 (显示 {display_weights.shape[0]}x{display_weights.shape[1]} / {in_features}x{out_features})')
        else:
            plt.imshow(weights, cmap='coolwarm', aspect='auto')
        
        plt.colorbar(label='权重值')
        plt.xlabel('输出神经元索引')
        plt.ylabel('输入神经元索引')
        
        # 保存图像
        save_path = os.path.join(save_dir, f'fc_weights_layer_{layer_idx+1}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"已保存第{layer_idx+1}层全连接层权重可视化到: {save_path}")
        
        # 对于连接到输出层的全连接层，可视化每个数字的权重模式
        if out_features == 10 and layer_idx == len(model.layers) - 1:
            plt.figure(figsize=(15, 8))
            plt.suptitle(f'第{layer_idx+1}层 (输出层) 每个数字的权重模式', fontsize=16)
            
            # 每个输出神经元对应一个数字
            for digit in range(10):
                plt.subplot(2, 5, digit + 1)
                
                # 如果是从展平层来的，尝试将权重重塑回图像形状
                if in_features == 12544:  # 64 * 14 * 14，可能来自残差模型
                    # 尝试将权重重塑为图像形状
                    weight_img = weights[:, digit].reshape(64, 14, 14)
                    # 取平均值，获得单通道图像
                    weight_img = np.mean(weight_img, axis=0)
                elif in_features == 3136:  # 64 * 7 * 7，可能来自标准CNN
                    weight_img = weights[:, digit].reshape(64, 7, 7)
                    weight_img = np.mean(weight_img, axis=0)
                elif in_features == 784:  # 直接从原始图像 (28*28)
                    weight_img = weights[:, digit].reshape(28, 28)
                else:
                    # 如果不能重塑为图像，显示条形图
                    plt.bar(range(min(50, in_features)), weights[:min(50, in_features), digit])
                    plt.title(f'数字 {digit}')
                    continue
                
                # 显示图像
                plt.imshow(weight_img, cmap='viridis')
                plt.title(f'数字 {digit}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # 为标题留出空间
            
            # 保存图像
            save_path = os.path.join(save_dir, f'output_weights_digit_patterns.png')
            plt.savefig(save_path)
            plt.close()
            
            print(f"已保存输出层数字模式到: {save_path}")

def visualize_model_gradients(model, X_test, y_test, sample_idx=0, save_dir='gradient_visualizations'):
    """可视化模型在特定样本上的梯度"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 准备单个样本
    if len(X_test.shape) == 2:  # (N, 784)
        X_sample = X_test[sample_idx:sample_idx+1].reshape(1, 1, 28, 28)
    else:  # (N, C, H, W)
        X_sample = X_test[sample_idx:sample_idx+1]
    
    y_sample = y_test[sample_idx:sample_idx+1]
    
    # 前向传播
    outputs = model.forward(X_sample, train=True)
    
    # 计算损失和梯度
    if hasattr(model, 'loss_function') and model.loss_function is not None:
        # 如果模型有损失函数，使用它
        loss = model.loss_function.forward(outputs, y_sample)
        grad = model.loss_function.backward()
    else:
        # 否则，手动创建交叉熵损失
        loss_function = CrossEntropyLoss()
        loss = loss_function.forward(outputs, y_sample)
        grad = loss_function.backward()
    
    # 反向传播
    model.backward(grad)
    
    # 收集每层的梯度
    gradients = []
    layer_names = []
    
    # 对于每一层，收集并可视化梯度
    for i, layer in enumerate(model.layers):
        layer_name = f"{i+1}: {layer.__class__.__name__}"
        
        if hasattr(layer, 'grads') and layer.grads:
            for param_name, param_grad in layer.grads.items():
                # 计算梯度的范数（L2范数）
                grad_norm = np.linalg.norm(param_grad)
                gradients.append(grad_norm)
                grad_name = f"{layer_name}.{param_name}"
                layer_names.append(grad_name)
                
                print(f"层 {grad_name} 梯度范数: {grad_norm:.6f}")
                
                # 对于卷积层权重，可视化梯度
                if isinstance(layer, Conv2D) and param_name == 'W':
                    plt.figure(figsize=(12, 10))
                    plt.suptitle(f'层 {grad_name} 梯度', fontsize=16)
                    
                    # 获取输出通道数
                    out_channels = param_grad.shape[0]
                    
                    # 确定布局
                    n_rows = int(np.ceil(np.sqrt(min(64, out_channels))))
                    n_cols = int(np.ceil(min(64, out_channels) / n_rows))
                    
                    # 显示每个输出通道的第一个输入通道的梯度
                    for j in range(min(64, out_channels)):
                        plt.subplot(n_rows, n_cols, j + 1)
                        
                        # 获取该滤波器的梯度
                        filter_grad = param_grad[j, 0]
                        
                        # 归一化以便可视化
                        if np.max(filter_grad) != np.min(filter_grad):
                            filter_grad = (filter_grad - np.min(filter_grad)) / (np.max(filter_grad) - np.min(filter_grad))
                        
                        plt.imshow(filter_grad, cmap='coolwarm')
                        plt.axis('off')
                    
                    # 保存图像
                    save_path = os.path.join(save_dir, f'gradient_{i}_{layer_name.replace(":", "_")}_{param_name}.png')
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    
                    print(f"已保存层 {grad_name} 的梯度可视化到: {save_path}")
    
    # 绘制所有层梯度范数的条形图
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(gradients)), gradients)
    plt.xticks(range(len(gradients)), layer_names, rotation=90)
    plt.title('各层参数梯度范数')
    plt.xlabel('层参数')
    plt.ylabel('梯度范数')
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'gradient_norms.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"已保存梯度范数可视化到: {save_path}")

def visualize_wrong_predictions(model, X_test, y_test, max_samples=10, save_dir='wrong_predictions'):
    """可视化模型错误预测的样本"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 确保数据格式正确
    if len(X_test.shape) == 2:  # (N, 784)
        X_cnn = X_test.reshape(-1, 1, 28, 28)
    else:
        X_cnn = X_test
    
    # 获取预测结果
    predictions = model.predict(X_cnn)
    
    # 找出错误预测的样本
    wrong_indices = np.where(predictions != y_test)[0]
    
    if len(wrong_indices) == 0:
        print("模型在测试集上没有错误预测！")
        return
    
    print(f"模型在测试集上有 {len(wrong_indices)} 个错误预测 (错误率: {len(wrong_indices) / len(y_test):.2%})")
    
    # 选择一部分错误样本进行可视化
    if len(wrong_indices) > max_samples:
        selected_indices = np.random.choice(wrong_indices, max_samples, replace=False)
    else:
        selected_indices = wrong_indices
    
    # 可视化每个错误预测的样本
    for i, idx in enumerate(selected_indices):
        # 获取样本、真实标签和预测
        sample = X_cnn[idx:idx+1]
        true_label = y_test[idx]
        pred_label = predictions[idx]
        
        # 获取模型输出
        outputs = model.forward(sample, train=False)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 显示图像
        plt.subplot(2, 2, 1)
        plt.imshow(sample[0, 0], cmap='gray')
        plt.title(f'真实标签: {true_label}')
        plt.axis('off')
        
        # 显示模型输出
        plt.subplot(2, 2, 2)
        bars = plt.bar(range(10), outputs[0])
        plt.title('模型输出')
        plt.xlabel('数字类别')
        plt.ylabel('激活值')
        plt.xticks(range(10))
        
        # 高亮真实标签和预测标签
        bars[true_label].set_color('green')
        bars[pred_label].set_color('red')
        
        # 显示预测摘要
        plt.subplot(2, 1, 2)
        confidence = outputs[0][pred_label] / np.sum(outputs[0])
        plt.text(0.5, 0.5, 
                 f"样本索引: {idx}\n"
                 f"真实标签: {true_label}\n"
                 f"预测标签: {pred_label}\n"
                 f"预测置信度: {confidence:.4f}",
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=14,
                 transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'wrong_pred_{i}_true_{true_label}_pred_{pred_label}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"已保存错误预测样本 {i+1}/{len(selected_indices)} (索引: {idx}) 到: {save_path}")

def main():
    """主函数"""
    print("MNIST神经网络模型可视化工具")
    print("=" * 40)
    
    # 加载测试数据
    print("\n加载MNIST测试数据...")
    X_test, y_test = load_mnist()
    if X_test is None:
        print("无法加载测试数据，请确保数据文件在正确的位置。")
        return
    
    # 查找可用的模型文件
    model_files = find_model_files()
    if not model_files:
        print("没有找到可用的模型文件，请确保模型保存在'models'目录中。")
        return
    
    # 让用户选择模型
    print("\n请选择要可视化的模型:")
    for i, path in enumerate(model_files, 1):
        print(f"{i}. {os.path.basename(path)}")
    
    try:
        choice = int(input("\n请输入模型编号 (1-{}): ".format(len(model_files))))
        if 1 <= choice <= len(model_files):
            selected_model = model_files[choice-1]
        else:
            print(f"无效的选择，请输入1-{len(model_files)}之间的数字")
            return
    except ValueError:
        print("请输入有效的数字")
        return
    
    # 加载选中的模型
    print(f"\n加载模型 {os.path.basename(selected_model)}...")
    model = load_model(selected_model)
    if model is None:
        print("无法加载模型，请检查模型文件是否损坏。")
        return
    
    # 创建模型名称（用于保存可视化结果）
    model_name = os.path.basename(selected_model).split('.')[0]
    base_dir = f'visualizations_{model_name}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 显示可视化选项
    print("\n可视化选项:")
    print("1. 可视化模型每一层（在单个样本上）")
    print("2. 可视化卷积层滤波器")
    print("3. 可视化全连接层权重")
    print("4. 可视化模型梯度")
    print("5. 可视化错误预测样本")
    print("6. 执行所有可视化")
    print("7. 退出")
    
    vis_choice = input("\n请选择可视化类型 (1-7): ")
    
    if vis_choice == '1':
        # 可视化模型每一层
        sample_idx = int(input("请输入要可视化的样本索引 (0-9999): "))
        if 0 <= sample_idx < len(X_test):
            layer_dir = os.path.join(base_dir, 'layers')
            print(f"\n正在可视化模型每一层（样本 {sample_idx}）...")
            visualize_all_layers(model, X_test, sample_idx, save_dir=layer_dir)
        else:
            print(f"无效的样本索引，必须在0-{len(X_test)-1}之间")
    
    elif vis_choice == '2':
        # 可视化卷积层滤波器
        filter_dir = os.path.join(base_dir, 'filters')
        print("\n正在可视化卷积层滤波器...")
        visualize_weight_filters(model, save_dir=filter_dir)
    
    elif vis_choice == '3':
        # 可视化全连接层权重
        fc_dir = os.path.join(base_dir, 'fc_weights')
        print("\n正在可视化全连接层权重...")
        visualize_fc_weights(model, save_dir=fc_dir)
    
    elif vis_choice == '4':
        # 可视化模型梯度
        sample_idx = int(input("请输入要可视化梯度的样本索引 (0-9999): "))
        if 0 <= sample_idx < len(X_test):
            grad_dir = os.path.join(base_dir, 'gradients')
            print(f"\n正在可视化模型梯度（样本 {sample_idx}）...")
            visualize_model_gradients(model, X_test, y_test, sample_idx, save_dir=grad_dir)
        else:
            print(f"无效的样本索引，必须在0-{len(X_test)-1}之间")
    
    elif vis_choice == '5':
        # 可视化错误预测样本
        wrong_dir = os.path.join(base_dir, 'wrong_predictions')
        max_samples = int(input("要可视化多少个错误预测样本? "))
        print(f"\n正在可视化最多 {max_samples} 个错误预测样本...")
        visualize_wrong_predictions(model, X_test, y_test, max_samples, save_dir=wrong_dir)
    
    elif vis_choice == '6':
        # 执行所有可视化
        print("\n正在执行所有可视化...")
        
        # 随机选择一个样本
        sample_idx = np.random.randint(0, len(X_test))
        print(f"随机选择样本索引: {sample_idx}")
        
        # 1. 可视化模型每一层
        layer_dir = os.path.join(base_dir, 'layers')
        print(f"\n正在可视化模型每一层（样本 {sample_idx}）...")
        visualize_all_layers(model, X_test, sample_idx, save_dir=layer_dir)
        
        # 2. 可视化卷积层滤波器
        filter_dir = os.path.join(base_dir, 'filters')
        print("\n正在可视化卷积层滤波器...")
        visualize_weight_filters(model, save_dir=filter_dir)
        
        # 3. 可视化全连接层权重
        fc_dir = os.path.join(base_dir, 'fc_weights')
        print("\n正在可视化全连接层权重...")
        visualize_fc_weights(model, save_dir=fc_dir)
        
        # 4. 可视化模型梯度
        grad_dir = os.path.join(base_dir, 'gradients')
        print(f"\n正在可视化模型梯度（样本 {sample_idx}）...")
        visualize_model_gradients(model, X_test, y_test, sample_idx, save_dir=grad_dir)
        
        # 5. 可视化错误预测样本
        wrong_dir = os.path.join(base_dir, 'wrong_predictions')
        print("\n正在可视化最多10个错误预测样本...")
        visualize_wrong_predictions(model, X_test, y_test, max_samples=10, save_dir=wrong_dir)
    
    elif vis_choice == '7':
        print("\n退出可视化工具...")
        return
    
    else:
        print("\n无效的选择")
        return
    
    print(f"\n所有可视化结果已保存到目录: {base_dir}")
    print("可视化完成！")

if __name__ == "__main__":
    main()