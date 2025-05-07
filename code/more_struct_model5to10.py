import numpy as np
from struct import unpack
import gzip
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import copy

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Xavier initialization for better convergence
        self.params['W'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.params['b'] = np.zeros(out_dim)
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params['W'] + self.params['b']
    
    def backward(self, grad):
        # Calculate gradients
        self.grads['W'] = self.inputs.T @ grad
        self.grads['b'] = np.sum(grad, axis=0)
        # Return gradient for next layer
        return grad @ self.params['W'].T

class LinearWithInit(Linear):
    def __init__(self, in_dim, out_dim, init_type='xavier'):
        super().__init__(in_dim, out_dim)
        
        if init_type == 'he':
            # He initialization
            self.params['W'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        elif init_type == 'xavier':
            # Xavier initialization
            self.params['W'] = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None
        self.input_shape = None
        
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.mask = inputs > 0
        return inputs * self.mask
    
    def backward(self, grad):
        # 确保梯度和掩码具有相同的形状
        if grad.shape != self.mask.shape:
            print(f"警告：ReLU层梯度形状 {grad.shape} 与掩码形状 {self.mask.shape} 不匹配")
            
            # 调整掩码以匹配梯度的形状
            if len(grad.shape) == len(self.mask.shape):
                # 裁剪掩码以匹配梯度形状
                slices = tuple(slice(0, dim) for dim in grad.shape)
                mask_cropped = self.mask[slices]
                
                print(f"已调整掩码为梯度形状: {grad.shape}")
                return grad * mask_cropped
        
        return grad * self.mask

class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.params['gamma'] = np.ones(num_features)
        self.params['beta'] = np.zeros(num_features)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.input_shape = None
        self.batch_mean = None
        self.batch_var = None
        self.x_norm = None
        self.inputs = None
    
    def forward(self, inputs, train=True):
        self.inputs = inputs
        self.input_shape = inputs.shape
        
        # Handle different input dimensions
        if len(inputs.shape) == 4:  # (N, C, H, W) for CNN
            N, C, H, W = inputs.shape
            x_reshaped = inputs.transpose(0, 2, 3, 1).reshape(-1, C)
        else:  # (N, D) for MLP
            x_reshaped = inputs
        
        if train:
            # Calculate batch statistics
            self.batch_mean = np.mean(x_reshaped, axis=0)
            self.batch_var = np.var(x_reshaped, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.x_norm = (x_reshaped - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            out = self.params['gamma'] * self.x_norm + self.params['beta']
        else:
            # Use running statistics for inference
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.params['gamma'] * x_norm + self.params['beta']
        
        # Reshape back to original shape
        if len(self.input_shape) == 4:
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return out
    
    def backward(self, grad):
        if len(self.input_shape) == 4:  # (N, C, H, W)
            N, C, H, W = self.input_shape
            
            # 确保梯度形状与输入形状匹配
            if grad.shape[2] != H or grad.shape[3] != W:
                print(f"警告：BatchNorm层收到的梯度形状 {grad.shape} 与输入形状 {self.input_shape} 不匹配")
                # 裁剪梯度或填充为0以匹配输入形状
                new_grad = np.zeros(self.input_shape)
                min_h = min(grad.shape[2], H)
                min_w = min(grad.shape[3], W)
                new_grad[:, :, :min_h, :min_w] = grad[:, :, :min_h, :min_w]
                grad = new_grad
                print(f"已调整梯度形状为: {grad.shape}")
            
            grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, C)
            x_reshaped = self.inputs.transpose(0, 2, 3, 1).reshape(-1, C)
        else:  # (N, D)
            grad_reshaped = grad
            x_reshaped = self.inputs
            
        m = x_reshaped.shape[0]
        
        # Calculate gradients
        dx_norm = grad_reshaped * self.params['gamma']
        
        # Backprop through the normalization
        dvar = np.sum(dx_norm * (x_reshaped - self.batch_mean) * (-0.5) * np.power(self.batch_var + self.eps, -1.5), axis=0)
        dmean = np.sum(dx_norm * (-1) / np.sqrt(self.batch_var + self.eps), axis=0) + dvar * np.sum(-2 * (x_reshaped - self.batch_mean), axis=0) / m
        
        dx = dx_norm / np.sqrt(self.batch_var + self.eps) + dvar * 2 * (x_reshaped - self.batch_mean) / m + dmean / m
        
        # Parameter gradients
        self.grads['gamma'] = np.sum(grad_reshaped * self.x_norm, axis=0)
        self.grads['beta'] = np.sum(grad_reshaped, axis=0)
        
        # Reshape back to original shape
        if len(self.input_shape) == 4:
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return dx

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with Xavier initialization
        weight_scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.params['W'] = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) * weight_scale
        self.params['b'] = np.zeros(out_channels)
        
        self.inputs = None
        self.padded = None
        self.output_shape = None
        self.x_col = None  # 缓存im2col结果用于反向传播
    
    def im2col(self, x, h_out, w_out):
        """Transform a batch of images into columns for efficient convolution."""
        N, C, H, W = x.shape
        k = self.kernel_size
        
        # Initialize output columns
        cols = np.zeros((N, C, k, k, h_out, w_out))
        
        # Fill the columns
        for h in range(h_out):
            h_start = h * self.stride
            for w in range(w_out):
                w_start = w * self.stride
                cols[:, :, :, :, h, w] = x[:, :, h_start:h_start+k, w_start:w_start+k]
        
        # Reshape to (N, h_out, w_out, C*k*k)
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * h_out * w_out, -1)
        return cols
    
    def forward(self, inputs):
        N, C, H, W = inputs.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        # 保存输入用于反向传播
        self.inputs = inputs
        
        # Apply padding if needed
        if p > 0:
            self.padded = np.pad(inputs, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        else:
            self.padded = inputs
            
        # Calculate output dimensions
        h_out = (self.padded.shape[2] - k) // s + 1
        w_out = (self.padded.shape[3] - k) // s + 1
        self.output_shape = (N, self.out_channels, h_out, w_out)
        
        # Prepare weights for computation
        W_col = self.params['W'].reshape(self.out_channels, -1)
        
        # Convert input to columns
        self.x_col = self.im2col(self.padded, h_out, w_out)
        
        # Perform convolution as matrix multiplication
        output = (W_col @ self.x_col.T).T
        output = output.reshape(N, h_out, w_out, self.out_channels).transpose(0, 3, 1, 2)
        
        # Add bias
        output += self.params['b'].reshape(1, self.out_channels, 1, 1)
        
        return output
    
    def col2im(self, cols, x_shape):
        """将列数据转换回图像格式
        
        Args:
            cols: 要转换回图像格式的列数据
            x_shape: 原始图像形状 (N, C, H, W)
        """
        N, C, H, W = x_shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        # 计算输出尺寸
        H_padded, W_padded = H + 2 * p, W + 2 * p
        h_out = (H_padded - k) // s + 1
        w_out = (W_padded - k) // s + 1
        
        # 创建填充后的数组
        x_padded = np.zeros((N, C, H_padded, W_padded))
        
        # 重塑列数据
        try:
            cols_reshaped = cols.reshape(N, h_out, w_out, C, k, k).transpose(0, 3, 4, 5, 1, 2)
            
            # 在填充数组中累积值
            for h in range(h_out):
                h_start = h * s
                for w in range(w_out):
                    w_start = w * s
                    x_padded[:, :, h_start:h_start+k, w_start:w_start+k] += cols_reshaped[:, :, :, :, h, w]
            
        except Exception as e:
            print(f"错误：col2im失败 - {e}")
            print("使用零梯度作为应急方案")
            
        # 去除填充（如果需要）
        if p > 0:
            return x_padded[:, :, p:-p, p:-p]
        return x_padded
    
    def backward(self, grad):
        """反向传播过程"""
        N, C, H, W = self.inputs.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        # 确保梯度形状与前向传播的输出形状匹配
        if grad.shape != self.output_shape:
            print(f"警告：卷积层反向传播收到的梯度形状 {grad.shape} 与期望形状 {self.output_shape} 不匹配")
            
            # 创建新的梯度，大小与期望的输出形状一致
            new_grad = np.zeros(self.output_shape)
            
            # 复制共同部分
            min_n = min(grad.shape[0], self.output_shape[0])
            min_c = min(grad.shape[1], self.output_shape[1])
            min_h = min(grad.shape[2], self.output_shape[2])
            min_w = min(grad.shape[3], self.output_shape[3])
            
            new_grad[:min_n, :min_c, :min_h, :min_w] = grad[:min_n, :min_c, :min_h, :min_w]
            grad = new_grad
            print(f"已调整梯度形状为: {self.output_shape}")
        
        # 重塑梯度用于计算
        h_out, w_out = self.output_shape[2], self.output_shape[3]
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # 计算权重和偏置的梯度
        self.grads['W'] = (grad_reshaped.T @ self.x_col).reshape(self.params['W'].shape)
        self.grads['b'] = np.sum(grad_reshaped, axis=0)
        
        # 计算下一层的梯度
        W_col = self.params['W'].reshape(self.out_channels, -1)
        dx_col = grad_reshaped @ W_col
        
        # 将列梯度转换回图像格式
        dx_padded = self.col2im(dx_col, (N, C, H, W))
        
        # 去除填充（如果有）
        if p > 0:
            dx = dx_padded[:, :, p:-p, p:-p]
        else:
            dx = dx_padded
            
        return dx

class Conv2DWithInit(Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_type='xavier'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        
        if init_type == 'he':
            # He initialization
            fan_in = in_channels * kernel_size * kernel_size
            self.params['W'] = np.random.randn(
                out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        elif init_type == 'xavier':
            # Xavier initialization
            fan_in = in_channels * kernel_size * kernel_size
            fan_out = out_channels * kernel_size * kernel_size
            self.params['W'] = np.random.randn(
                out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (fan_in + fan_out))

class SimpleResidualBlock(Layer):
    """简化版残差块：确保主路径和跳跃连接的尺寸一致"""
    def __init__(self, in_channels, out_channels, use_bn=False, apply_pool=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.apply_pool = apply_pool
        
        # 强制使用padding=1确保卷积不改变空间尺寸
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        
        # 缓存中间结果
        self.cache = {}
        
        if use_bn:
            self.bn1 = BatchNorm(out_channels)
            self.bn2 = BatchNorm(out_channels)
        
        # 为不同输入/输出通道数创建1x1卷积的跳跃连接
        if in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if use_bn:
                self.shortcut_bn = BatchNorm(out_channels)
        else:
            self.shortcut = None

        # 根据需要创建池化层
        if apply_pool:
            self.pool = MaxPool2D(2)
        else:
            self.pool = None

    def forward(self, x, train=True):
        # 保存输入
        self.cache['input'] = x.copy()
        
        # 将原始输入保存为恒等映射的基础
        identity = x
        
        # 第一个卷积块
        out = self.conv1.forward(x)
        self.cache['conv1_out'] = out.copy()
        
        if self.use_bn:
            out = self.bn1.forward(out, train)
            self.cache['bn1_out'] = out.copy()
            
        out = self.relu1.forward(out)
        self.cache['relu1_out'] = out.copy()

        # 第二个卷积块
        out = self.conv2.forward(out)
        self.cache['conv2_out'] = out.copy()
        
        if self.use_bn:
            out = self.bn2.forward(out, train)
            self.cache['bn2_out'] = out.copy()

        # 处理跳跃连接
        if self.shortcut is not None:
            identity = self.shortcut.forward(x)
            self.cache['shortcut_out'] = identity.copy()
            
            if self.use_bn:
                identity = self.shortcut_bn.forward(identity, train)
                self.cache['shortcut_bn_out'] = identity.copy()
        
        # 确保恒等映射与主路径的尺寸一致
        if identity.shape != out.shape:
            print(f"形状不匹配：恒等映射 {identity.shape} vs 主路径 {out.shape}")
            
            # 确定共同尺寸
            min_n = min(identity.shape[0], out.shape[0])
            min_c = min(identity.shape[1], out.shape[1])
            min_h = min(identity.shape[2], out.shape[2])
            min_w = min(identity.shape[3], out.shape[3])
            
            # 裁剪到共同尺寸
            identity = identity[:min_n, :min_c, :min_h, :min_w]
            out = out[:min_n, :min_c, :min_h, :min_w]
            print(f"已调整为共同形状: {out.shape}")

        # 执行残差连接
        out = out + identity
        self.cache['pre_relu2'] = out.copy()
        
        out = self.relu2.forward(out)
        self.cache['relu2_out'] = out.copy()

        # 应用池化（如果指定）
        if self.pool:
            out = self.pool.forward(out)
            self.cache['pool_out'] = out.copy()

        return out

    def backward(self, dout):
        # 保存原始梯度形状用于调试
        orig_dout_shape = dout.shape
        
        # 池化层的反向传播
        if self.pool:
            dout = self.pool.backward(dout)
        
        # 第二个ReLU层的反向传播
        dout = self.relu2.backward(dout)
        
        # 获取残差连接之前的输出形状
        pre_relu2_shape = self.cache['pre_relu2'].shape
        
        # 确保梯度尺寸与pre_relu2一致
        if dout.shape != pre_relu2_shape:
            print(f"警告：残差块中梯度形状 {dout.shape} 与pre_relu2形状 {pre_relu2_shape} 不匹配")
            
            # 调整梯度形状
            new_dout = np.zeros(pre_relu2_shape)
            min_n = min(dout.shape[0], pre_relu2_shape[0])
            min_c = min(dout.shape[1], pre_relu2_shape[1])
            min_h = min(dout.shape[2], pre_relu2_shape[2])
            min_w = min(dout.shape[3], pre_relu2_shape[3])
            
            new_dout[:min_n, :min_c, :min_h, :min_w] = dout[:min_n, :min_c, :min_h, :min_w]
            dout = new_dout
            print(f"已调整梯度形状为: {pre_relu2_shape}")
        
        # 分流梯度：一部分给恒等连接，一部分给主路径
        didentity = dout.copy()
        dmain = dout.copy()
        
        # 主路径的反向传播
        if self.use_bn:
            dmain = self.bn2.backward(dmain)
        
        dmain = self.conv2.backward(dmain)
        dmain = self.relu1.backward(dmain)
        
        if self.use_bn:
            dmain = self.bn1.backward(dmain)
        
        dmain = self.conv1.backward(dmain)

        # 跳跃连接的反向传播
        if self.shortcut is not None:
            if self.use_bn:
                didentity = self.shortcut_bn.backward(didentity)
            didentity = self.shortcut.backward(didentity)
        
        # 检查并调整梯度形状，确保它们可以安全地相加
        input_shape = self.cache['input'].shape
        
        if dmain.shape != input_shape:
            print(f"警告：主路径梯度形状 {dmain.shape} 与输入形状 {input_shape} 不匹配")
            
            # 创建正确形状的梯度
            new_dmain = np.zeros(input_shape)
            min_n = min(dmain.shape[0], input_shape[0])
            min_c = min(dmain.shape[1], input_shape[1])
            min_h = min(dmain.shape[2], input_shape[2])
            min_w = min(dmain.shape[3], input_shape[3])
            
            new_dmain[:min_n, :min_c, :min_h, :min_w] = dmain[:min_n, :min_c, :min_h, :min_w]
            dmain = new_dmain
            print(f"已调整主路径梯度形状为: {input_shape}")
        
        if didentity.shape != input_shape:
            print(f"警告：恒等映射梯度形状 {didentity.shape} 与输入形状 {input_shape} 不匹配")
            
            # 创建正确形状的梯度
            new_didentity = np.zeros(input_shape)
            min_n = min(didentity.shape[0], input_shape[0])
            min_c = min(didentity.shape[1], input_shape[1])
            min_h = min(didentity.shape[2], input_shape[2])
            min_w = min(didentity.shape[3], input_shape[3])
            
            new_didentity[:min_n, :min_c, :min_h, :min_w] = didentity[:min_n, :min_c, :min_h, :min_w]
            didentity = new_didentity
            print(f"已调整恒等映射梯度形状为: {input_shape}")
        
        # 返回合并后的梯度
        return dmain + didentity

class MaxPool2D(Layer):
    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = pool_size if stride is None else stride
        self.inputs = None
        self.mask = None
        self.output_shape = None
    
    def forward(self, inputs):
        """前向传播，进行最大池化操作"""
        self.inputs = inputs
        N, C, H, W = inputs.shape
        pool = self.pool_size
        stride = self.stride
        
        # 检查输入图像是否太小
        if H < pool or W < pool:
            print(f"警告: 输入图像尺寸 {H}x{W} 小于池化窗口 {pool}x{pool}")
            # 如果输入太小，进行padding或调整池化窗口大小
            if H < pool:
                padded_h = np.zeros((N, C, pool - H, W))
                inputs = np.concatenate([inputs, padded_h], axis=2)
                H = pool
            if W < pool:
                padded_w = np.zeros((N, C, H, pool - W))
                inputs = np.concatenate([inputs, padded_w], axis=3)
                W = pool
            print(f"已调整输入尺寸为: {inputs.shape}")
            self.inputs = inputs
        
        # 计算输出维度
        h_out = (H - pool) // stride + 1
        w_out = (W - pool) // stride + 1
        
        # 如果计算出的尺寸小于1，则强制设为1
        h_out = max(1, h_out)
        w_out = max(1, w_out)
        
        self.output_shape = (N, C, h_out, w_out)
        
        # 初始化输出数组和掩码
        output = np.zeros(self.output_shape)
        self.mask = np.zeros_like(inputs)
        
        # 执行最大池化
        for n in range(N):
            for c in range(C):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * stride
                        w_start = w * stride
                        
                        # 确保不超出边界
                        h_end = min(h_start + pool, H)
                        w_end = min(w_start + pool, W)
                        
                        # 选择区域
                        patch = inputs[n, c, h_start:h_end, w_start:w_end]
                        
                        # 处理空patch的情况
                        if patch.size == 0:
                            print(f"警告: 遇到空patch在位置 n={n}, c={c}, h={h}, w={w}")
                            continue
                        
                        # 找到最大值及其位置
                        max_val = np.max(patch)
                        max_idx = np.argmax(patch.flatten())
                        
                        # 转换1D索引为2D索引
                        patch_h, patch_w = h_end - h_start, w_end - w_start
                        max_h, max_w = np.unravel_index(max_idx, (patch_h, patch_w))
                        
                        # 在输出存储最大值
                        output[n, c, h, w] = max_val
                        
                        # 保存位置用于反向传播
                        self.mask[n, c, h_start + max_h, w_start + max_w] = 1
        
        return output
    
    def backward(self, grad):
        """反向传播，分配梯度到最大值位置"""
        N, C, H_out, W_out = grad.shape
        dx = np.zeros_like(self.inputs)
        
        # 确保梯度形状与输出形状一致
        if grad.shape != self.output_shape:
            print(f"警告：MaxPool2D层收到的梯度形状 {grad.shape} 与期望形状 {self.output_shape} 不匹配")
            
            # 调整梯度形状
            new_grad = np.zeros(self.output_shape)
            min_n = min(grad.shape[0], self.output_shape[0])
            min_c = min(grad.shape[1], self.output_shape[1])
            min_h = min(grad.shape[2], self.output_shape[2])
            min_w = min(grad.shape[3], self.output_shape[3])
            
            new_grad[:min_n, :min_c, :min_h, :min_w] = grad[:min_n, :min_c, :min_h, :min_w]
            grad = new_grad
            print(f"已调整梯度形状为: {self.output_shape}")
            
            # 更新输出尺寸以匹配调整后的梯度
            H_out, W_out = self.output_shape[2], self.output_shape[3]
        
        # 分配梯度
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # 确保不会越界
                        h_end = min(h_start + self.pool_size, self.inputs.shape[2])
                        w_end = min(w_start + self.pool_size, self.inputs.shape[3])
                        
                        # 确保区域有效
                        if h_start >= h_end or w_start >= w_end:
                            continue
                        
                        # 分配梯度到最大值位置
                        mask_patch = self.mask[n, c, h_start:h_end, w_start:w_end]
                        dx[n, c, h_start:h_end, w_start:w_end] += mask_patch * grad[n, c, h, w]
        
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

class Softmax:
    def __init__(self):
        self.probs = None
    
    def forward(self, inputs):
        # For numerical stability
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
        
        # Calculate cross entropy loss
        N = inputs.shape[0]
        loss = -np.sum(np.log(self.probs[np.arange(N), labels] + 1e-10)) / N  # 添加小数值避免log(0)
        return loss
    
    def backward(self):
        # Gradient of cross entropy loss
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        grad /= N
        return grad

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        # Higher score is better (negative loss)
        score = -val_loss
        
        if self.best_loss is None:
            self.best_loss = score
            self.best_model = copy.deepcopy(model)
        elif score < self.best_loss + self.min_delta:
            # Validation loss didn't improve enough
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation loss improved
            self.best_loss = score
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        
        return self.early_stop

class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.dropout_layers = []
        self.t = 0  # Time step for Adam
        
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
        """Update weights using SGD with momentum or Adam"""
        # Initialize time step for Adam
        if optimizer == 'adam':
            self.t += 1
        
        # Update parameters for each layer
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for param in layer.params:
                    # Apply L2 regularization
                    if param == 'W':  # Only apply weight decay to weights, not biases
                        grad = layer.grads[param] + weight_decay * layer.params[param]
                    else:
                        grad = layer.grads[param]
                    
                    if optimizer == 'sgd':
                        # SGD with momentum
                        # Initialize momentum buffer if not exists
                        if f'v_{param}' not in layer.__dict__:
                            layer.__dict__[f'v_{param}'] = np.zeros_like(layer.params[param])
                        
                        # Update momentum
                        layer.__dict__[f'v_{param}'] = momentum * layer.__dict__[f'v_{param}'] - lr * grad
                        
                        # Update parameters
                        layer.params[param] += layer.__dict__[f'v_{param}']
                    
                    elif optimizer == 'adam':
                        # Initialize momentum and squared gradient buffers
                        if f'm_{param}' not in layer.__dict__:
                            layer.__dict__[f'm_{param}'] = np.zeros_like(layer.params[param])
                            layer.__dict__[f'v_{param}'] = np.zeros_like(layer.params[param])
                        
                        # Update biased first moment estimate
                        layer.__dict__[f'm_{param}'] = beta1 * layer.__dict__[f'm_{param}'] + (1 - beta1) * grad
                        
                        # Update biased second moment estimate
                        layer.__dict__[f'v_{param}'] = beta2 * layer.__dict__[f'v_{param}'] + (1 - beta2) * (grad ** 2)
                        
                        # Bias correction
                        m_corrected = layer.__dict__[f'm_{param}'] / (1 - beta1 ** self.t)
                        v_corrected = layer.__dict__[f'v_{param}'] / (1 - beta2 ** self.t)
                        
                        # Update parameters
                        layer.params[param] -= lr * m_corrected / (np.sqrt(v_corrected) + eps)
    
    def predict(self, X):
        outputs = self.forward(X, train=False)
        return np.argmax(outputs, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model on the given data and labels"""
        # Check that X and y have the same number of samples
        assert X.shape[0] == y.shape[0], f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}"
        
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
def create_model_directory():
    """Create model directory"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created model directory: models/")
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created plots directory: plots/")

def load_mnist():
    """Load MNIST dataset"""
    # Try loading from current directory
    try:
        # Load training data
        with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            X_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        
        with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            y_train = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Load test data
        with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        
        with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            y_test = np.frombuffer(f.read(), dtype=np.uint8)
    except FileNotFoundError:
        # If not found in current directory, try original path
        try:
            # Load training data
            with gzip.open('/home/PJ1/codes/dataset/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
                magic, num, rows, cols = unpack('>4I', f.read(16))
                X_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            
            with gzip.open('/home/PJ1/codes/dataset/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
                magic, num = unpack('>2I', f.read(8))
                y_train = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Load test data
            with gzip.open('/home/PJ1/codes/dataset/MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
                magic, num, rows, cols = unpack('>4I', f.read(16))
                X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            
            with gzip.open('/home/PJ1/codes/dataset/MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
                magic, num = unpack('>2I', f.read(8))
                y_test = np.frombuffer(f.read(), dtype=np.uint8)
        except FileNotFoundError:
            print("Could not find MNIST dataset. Please ensure the dataset is in the current directory or the specified path.")
            return None, None, None, None
    
    # Normalize data
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    return X_train, y_train, X_test, y_test

def save_model(model, filename):
    """Save complete model using pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load complete model using pickle"""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        
        # Print model structure for debugging
        print("Model structure:")
        for i, layer in enumerate(model.layers):
            layer_str = f"  Layer {i}: {layer.__class__.__name__}"
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    layer_str += f", {param_name} shape: {param.shape}"
            print(layer_str)
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def enhanced_data_augmentation(X, y, num_augmentations=1):
    """Enhanced data augmentation including shift, rotation, and scaling"""
    N, C, H, W = X.shape
    X_aug = []
    y_aug = []
    
    for i in range(N):
        # Original image
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Augmented images
        for _ in range(num_augmentations):
            img = X[i, 0].copy()  # Get image (remove channel dimension)
            
            # Randomly select augmentation type
            aug_type = np.random.choice(['shift', 'rotate', 'scale', 'combined'])
            
            if aug_type == 'shift' or aug_type == 'combined':
                # Random shift (up to 2 pixels)
                shift_h = np.random.randint(-2, 3)
                shift_w = np.random.randint(-2, 3)
                
                # Apply shift
                img_shifted = np.zeros_like(img)
                
                # Determine valid indices
                h_start_src = max(0, -shift_h)
                h_end_src = min(H, H - shift_h)
                w_start_src = max(0, -shift_w)
                w_end_src = min(W, W - shift_w)
                
                h_start_dst = max(0, shift_h)
                h_end_dst = min(H, H + shift_h)
                w_start_dst = max(0, shift_w)
                w_end_dst = min(W, W + shift_w)
                
                # Copy valid part
                img_shifted[h_start_dst:h_end_dst, w_start_dst:w_end_dst] = img[h_start_src:h_end_src, w_start_src:w_end_src]
                img = img_shifted
            
            if aug_type == 'rotate' or aug_type == 'combined':
                # Random rotation (max 10 degrees)
                angle = np.random.uniform(-10, 10) * np.pi / 180
                
                # Image center
                center_x, center_y = W // 2, H // 2
                
                # Rotation matrix
                cos_theta, sin_theta = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])
                
                # Create rotated image
                img_rotated = np.zeros_like(img)
                
                # Apply rotation
                for h in range(H):
                    for w in range(W):
                        # Coordinates relative to center
                        y_centered, x_centered = h - center_y, w - center_x
                        
                        # Rotate
                        new_x, new_y = rot_matrix @ [x_centered, y_centered]
                        
                        # Convert back to absolute coordinates
                        new_w, new_h = int(new_x + center_x), int(new_y + center_y)
                        
                        # Check if within original image boundaries
                        if 0 <= new_w < W and 0 <= new_h < H:
                            img_rotated[h, w] = img[new_h, new_w]
                
                img = img_rotated
            
            if aug_type == 'scale' or aug_type == 'combined':
                # Random scaling (0.9-1.1x)
                scale = np.random.uniform(0.9, 1.1)
                
                # Create scaled image
                img_scaled = np.zeros_like(img)
                
                # Image center
                center_x, center_y = W // 2, H // 2
                
                # Apply scaling
                for h in range(H):
                    for w in range(W):
                        # Calculate source coordinates
                        src_h = int((h - center_h) / scale + center_h)
                        src_w = int((w - center_w) / scale + center_w)
                        
                        # Check if within bounds
                        if 0 <= src_h < H and 0 <= src_w < W:
                            img_scaled[h, w] = img[src_h, src_w]
                
                img = img_scaled
            
            # Add augmented image
            X_aug.append(img.reshape(1, H, W))
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

def visualize_cnn_filters(model, layer_idx=0, model_name="cnn"):
    """Visualize filters of a convolutional layer"""
    # Find all convolutional layers
    conv_layers = [(i, layer) for i, layer in enumerate(model.layers) if isinstance(layer, Conv2D)]
    
    if not conv_layers:
        return
    
    # Use the specified layer index if valid, otherwise use the first conv layer
    if 0 <= layer_idx < len(conv_layers):
        idx, layer = conv_layers[layer_idx]
    else:
        idx, layer = conv_layers[0]
    
    # Get weights of the selected convolutional layer
    filters = layer.params['W']
    
    # Number of filters to display
    n_filters = min(32, filters.shape[0])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot each filter
    for i in range(n_filters):
        plt.subplot(4, 8, i + 1)
        
        # For filters with multiple input channels, average over channels
        if filters.shape[1] > 1:
            filter_img = np.mean(filters[i], axis=0)
        else:
            filter_img = filters[i, 0]
        
        plt.imshow(filter_img, cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f'CNN Filters from Layer {idx}')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_filters_layer{idx}.png')
    plt.close()

def train_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, 
               batch_size=64, epochs=50, lr=0.01, optimizer='sgd', 
               use_data_augmentation=False, num_augmentations=1,
               patience=5, verbose=True):
    """Train model with early stopping and various improvements"""
    # Create early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    
    loss_function = model.loss_function
    
    # Initialize training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Start timer
    start_time = time.time()
    
    # Train
    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}')
        
        # Prepare training data
        if use_data_augmentation:
            X_train_aug, y_train_aug = enhanced_data_augmentation(X_train, y_train, num_augmentations)
            if verbose:
                print(f'  Data augmentation: original {len(X_train)} samples, augmented to {len(X_train_aug)} samples')
        else:
            X_train_aug, y_train_aug = X_train, y_train
        
        # Shuffle data
        indices = np.random.permutation(len(X_train_aug))
        X_train_shuffled = X_train_aug[indices]
        y_train_shuffled = y_train_aug[indices]
        
        # Initialize loss and correct predictions
        epoch_loss = 0
        correct_predictions = 0
        
        # Train by batches
        for i in tqdm(range(0, len(X_train_shuffled), batch_size), desc=f'  Training progress'):
            # Get batch
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward pass
            outputs = model.forward(X_batch)
            loss = loss_function.forward(outputs, y_batch)
            epoch_loss += loss * len(X_batch)
            
            # Calculate training accuracy
            predictions = np.argmax(outputs, axis=1)
            correct_predictions += np.sum(predictions == y_batch)
            
            # Backward pass
            grad = loss_function.backward()
            model.backward(grad)
            
            # Update weights
            if optimizer == 'adam':
                model.update(lr, optimizer='adam')
            else:
                model.update(lr, optimizer='sgd', momentum=0.9, weight_decay=1e-4)
        
        # Calculate average loss and training accuracy
        epoch_loss /= len(X_train_shuffled)
        train_accuracy = correct_predictions / len(X_train_shuffled)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on validation set
        val_outputs = model.forward(X_val, train=False)
        val_loss = loss_function.forward(val_outputs, y_val)
        val_accuracy = model.evaluate(X_val, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if verbose:
            print(f'  Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Check early stopping
        if early_stopping(val_loss, model):
            if verbose:
                print(f'  Early stopping at epoch {epoch+1}')
            # Restore best model
            model = early_stopping.best_model
            break
        
        # Learning rate schedule
        if epoch > 0 and epoch % 5 == 0:
            lr *= 0.5
            if verbose:
                print(f'  Reduced learning rate to {lr:.6f}')
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Save final model
    save_model(model, f'models/{model_name}.pkl')
    
    # Evaluate on test set
    final_test_accuracy = model.evaluate(X_test, y_test)
    print(f'Final test accuracy: {final_test_accuracy:.4f}')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.axhline(y=final_test_accuracy, color='r', linestyle='--', label=f'Test ({final_test_accuracy:.4f})')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png')
    plt.close()
    
    # Visualize CNN filters
    if any(isinstance(layer, Conv2D) for layer in model.layers):
        visualize_cnn_filters(model, 0, model_name)
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_test_accuracy': final_test_accuracy,
        'training_time': training_time
    }

def create_model5():
    """Model 5: 深层CNN + 残差块"""
    model = Network()
    
    # 输入：1x28x28 → 卷积 → 32x28x28
    # 使用padding=1确保空间尺寸保持不变
    model.add(Conv2D(1, 32, 3, stride=1, padding=1))
    model.add(ReLU())

    # 残差块：32x28x28 → 64x28x28 → 池化后 64x14x14
    # 确保所有卷积都使用padding=1保持空间尺寸
    model.add(SimpleResidualBlock(32, 64, use_bn=False, apply_pool=True))

    # 展平 + 全连接层
    model.add(Flatten())  # → 64*14*14 = 12544
    model.add(Linear(64 * 14 * 14, 256))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(256, 10))

    model.set_loss(CrossEntropyLoss())
    return model


def create_model6():
    """Model 6: Model 5 + Batch Normalization"""
    model = Network()
    
    # 使用padding=1确保空间尺寸保持不变
    model.add(Conv2D(1, 32, 3, stride=1, padding=1))
    model.add(BatchNorm(32))
    model.add(ReLU())
    
    # 添加具有批归一化的简化残差块
    model.add(SimpleResidualBlock(32, 64, use_bn=True, apply_pool=True))
    
    # 全连接层
    model.add(Flatten())
    model.add(Linear(64 * 14 * 14, 256))  # 14x14是因为apply_pool=True将尺寸减半
    model.add(BatchNorm(256))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(256, 10))
    
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    return model

def create_model7():
    """Model 7: Model 6 with Adam optimizer (architecture same as Model 6)"""
    return create_model6()  # Adam will be specified during training

def create_model8():
    """Model 8: Model 6 with Data Augmentation (architecture same as Model 6)"""
    return create_model6()  # Data augmentation will be applied during training

def create_model9():
    """Model 9: Deep CNN with He Initialization"""
    model = Network()
    
    # First convolutional layer with He init
    model.add(Conv2DWithInit(1, 32, 5, padding=2, init_type='he'))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Second convolutional layer with He init
    model.add(Conv2DWithInit(32, 64, 3, padding=1, init_type='he'))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Fully connected layers with He init
    model.add(Flatten())
    model.add(LinearWithInit(64 * 7 * 7, 256, init_type='he'))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(LinearWithInit(256, 10, init_type='he'))
    
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    return model

def create_model10():
    """Model 10: Deep CNN with Xavier Initialization"""
    model = Network()
    
    # First convolutional layer with Xavier init
    model.add(Conv2DWithInit(1, 32, 5, padding=2, init_type='xavier'))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Second convolutional layer with Xavier init
    model.add(Conv2DWithInit(32, 64, 3, padding=1, init_type='xavier'))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Fully connected layers with Xavier init
    model.add(Flatten())
    model.add(LinearWithInit(64 * 7 * 7, 256, init_type='xavier'))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(LinearWithInit(256, 10, init_type='xavier'))
    
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    return model

def compare_models(models_results):
    """Compare all models"""
    # Create figure for comparison
    plt.figure(figsize=(12, 10))
    
    # Plot test accuracy comparison
    plt.subplot(2, 1, 1)
    names = list(models_results.keys())
    accuracies = [results['final_test_accuracy'] for results in models_results.values()]
    plt.bar(names, accuracies)
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.95, 1.0)  # Set a reasonable y-axis range
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    # Plot training time comparison
    plt.subplot(2, 1, 2)
    training_times = [results['training_time'] for results in models_results.values()]
    plt.bar(names, training_times)
    plt.title('Model Comparison - Training Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/models_comparison.png')
    plt.close()
    
    # Print comparison results
    print("\nModel Comparison Results")
    print("-" * 80)
    print(f"{'Model':<20} {'Test Accuracy':<15} {'Training Time':<15}")
    print("-" * 80)
    
    for name, results in models_results.items():
        print(f"{name:<20} {results['final_test_accuracy']:.4f}{'':<9} {results['training_time']:.2f}s")
    
    # Find best model
    best_model = max(models_results.items(), key=lambda x: x[1]['final_test_accuracy'])
    print("\nBest model: ", best_model[0])
    print(f"Test accuracy: {best_model[1]['final_test_accuracy']:.4f}")
    print(f"Training time: {best_model[1]['training_time']:.2f} seconds")

def main():
    """Main function to train and evaluate all models"""
    # Create model directory
    create_model_directory()
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    if X_train is None:
        return
    
    # Reshape data to (N, C, H, W) format for CNN
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Split training data into training and validation sets
    validation_size = 10000
    X_val = X_train[:validation_size]
    y_val = y_train[:validation_size]
    X_train = X_train[validation_size:]
    y_train = y_train[validation_size:]
    
    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    
    # Training parameters
    batch_size = 64
    max_epochs = 25  # Maximum epochs, early stopping may terminate sooner
    patience = 5  # Early stopping patience
    
    # Store results for comparison
    models_results = {}
    
    # Model 5: Deep CNN with Residual Block
    print("\n--- Training Model 5: Deep CNN with Residual Block ---")
    model5 = create_model5()
    _, results5 = train_model(
        model5, "model5_residual", X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, epochs=max_epochs, lr=0.01, patience=patience
    )
    models_results["Model 5: Residual"] = results5
    
    # Model 6: Model 5 + Batch Normalization
    print("\n--- Training Model 6: Model 5 + Batch Normalization ---")
    model6 = create_model6()
    _, results6 = train_model(
        model6, "model6_residual_bn", X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, epochs=max_epochs, lr=0.01, patience=patience
    )
    models_results["Model 6: Residual+BN"] = results6
    
    # Model 7: Model 6 with Adam optimizer
    print("\n--- Training Model 7: Model 6 with Adam Optimizer ---")
    model7 = create_model7()
    _, results7 = train_model(
        model7, "model7_residual_bn_adam", X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, epochs=max_epochs, lr=0.001, optimizer='adam', patience=patience
    )
    models_results["Model 7: Residual+BN+Adam"] = results7
    
    # Model 8: Model 6 with Data Augmentation
    print("\n--- Training Model 8: Model 6 with Data Augmentation ---")
    model8 = create_model8()
    _, results8 = train_model(
        model8, "model8_residual_bn_augment", X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, epochs=max_epochs, lr=0.01, 
        use_data_augmentation=True, num_augmentations=1, patience=patience
    )
    models_results["Model 8: Residual+BN+Augment"] = results8
    
    # Model 9: Deep CNN with He Initialization
    print("\n--- Training Model 9: Deep CNN with He Initialization ---")
    model9 = create_model9()
    _, results9 = train_model(
        model9, "model9_he_init", X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, epochs=max_epochs, lr=0.01, patience=patience
    )
    models_results["Model 9: He Init"] = results9
    
    # Model 10: Deep CNN with Xavier Initialization
    print("\n--- Training Model 10: Deep CNN with Xavier Initialization ---")
    model10 = create_model10()
    _, results10 = train_model(
        model10, "model10_xavier_init", X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, epochs=max_epochs, lr=0.01, patience=patience
    )
    models_results["Model 10: Xavier Init"] = results10
    
    # Compare all models
    compare_models(models_results)

if __name__ == "__main__":
    main()