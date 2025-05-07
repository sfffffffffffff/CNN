from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 记录输入用于反向传播
        # 线性变换: Y = X * W + b
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        
        # 计算W的梯度: dL/dW = X^T * dL/dY
        self.grads['W'] = np.dot(self.input.T, grad)
        
        # 如果使用权重衰减(L2正则化)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        # 计算b的梯度: dL/db = sum(dL/dY)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        
        # 计算传递给前一层的梯度: dL/dX = dL/dY * W^T
        dX = np.dot(grad, self.W.T)
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Optimized for better performance with unified parameter update.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.optimizable = True  # 标记为可优化层
        
        # 初始化权重和偏置，确保与其他层使用相同的参数命名
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        
        # 将参数统一存储在params字典中，与Linear层保持一致
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        
        self.input = None
        self.input_padded = None
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        # 预计算输出维度以提高效率
        self.out_h = None
        self.out_w = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def _pad(self, X):
        """
        为输入X添加填充
        X: [batch, channels, H, W]
        """
        if self.padding == 0:
            return X
            
        batch_size, channels, height, width = X.shape
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        
        X_padded = np.zeros((batch_size, channels, padded_height, padded_width))
        X_padded[:, :, self.padding:self.padding+height, self.padding:self.padding+width] = X
        
        return X_padded
    
    def forward(self, X):
        """
        使用im2col方法实现的高效卷积
        input X: [batch, channels, H, W]
        """
        self.input = X
        batch_size, in_channels, height, width = X.shape
        
        # 应用填充
        X_padded = self._pad(X)
        self.input_padded = X_padded
        
        # 计算输出维度
        self.out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # im2col转换
        # 将图像转换为列，以便可以使用矩阵乘法进行卷积
        col = self._im2col(X_padded, self.kernel_size, self.stride)
        
        # 重塑卷积核以便矩阵乘法
        W_col = self.W.reshape(self.out_channels, -1)
        
        # 执行卷积作为矩阵乘法
        out = np.matmul(W_col, col) + self.b.reshape(-1, 1)
        
        # 重塑输出
        out = out.reshape(self.out_channels, self.out_h, self.out_w, batch_size)
        out = out.transpose(3, 0, 1, 2)
        
        return out

    def _im2col(self, input_data, filter_h, stride=1):
        """
        将输入数据转换为列矩阵，用于快速卷积
        """
        N, C, H, W = input_data.shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_h) // stride + 1
        
        img = np.pad(input_data, [(0, 0), (0, 0), (0, 0), (0, 0)], 'constant')
        col = np.zeros((N, C, filter_h, filter_h, out_h, out_w))
        
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_h):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        
        col = col.transpose(1, 2, 3, 0, 4, 5).reshape(C * filter_h * filter_h, N * out_h * out_w)
        
        return col

    def backward(self, grads):
        """
        优化的反向传播，使用矩阵操作代替循环
        """
        batch_size, _, out_height, out_width = grads.shape
        _, _, in_height, in_width = self.input.shape
        
        # 初始化梯度
        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(self.out_channels,)
        dX_padded = np.zeros_like(self.input_padded)
        
        # 重塑梯度为适合矩阵乘法的形式
        grads_reshaped = grads.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        
        # 构建图像块矩阵
        img_cols = np.zeros((self.in_channels * self.kernel_size * self.kernel_size, 
                            batch_size * out_height * out_width))
        
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = w * self.stride
                    w_end = w_start + self.kernel_size
                    img_patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                    img_cols[:, b * out_height * out_width + h * out_width + w] = img_patch.reshape(-1)
        
        # 计算权重梯度
        dW = np.matmul(grads_reshaped, img_cols.T).reshape(self.out_channels, self.in_channels, 
                                                        self.kernel_size, self.kernel_size)
        
        # 计算输入梯度
        W_reshaped = self.W.reshape(self.out_channels, -1)
        dX_cols = np.matmul(W_reshaped.T, grads_reshaped)
        
        # 重塑回原始形状
        for b in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = w * self.stride
                    w_end = w_start + self.kernel_size
                    dX_padded[b, :, h_start:h_end, w_start:w_end] += dX_cols[:, b * out_height * out_width + h * out_width + w].reshape(
                        self.in_channels, self.kernel_size, self.kernel_size)
        
        # 如果有权重衰减，添加L2正则化梯度
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W
        
        # 移除输入梯度的填充
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:self.padding+in_height, self.padding:self.padding+in_width]
        else:
            dX = dX_padded
        
        return dX
    
    def clear_grad(self):
        """重置梯度 - 与其他层保持一致"""
        self.grads = {'W': None, 'b': None}
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        
        self.inputs = None
        self.batch_size = None
        self.probs = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels: [batch_size, ]
        """
        self.inputs = predicts
        self.batch_size = predicts.shape[0]
        self.labels = labels
        
        # 应用softmax，如果需要
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        
        # 确保索引有效
        if labels.max() >= self.probs.shape[1]:
            raise ValueError(f"标签最大值为{labels.max()}，但模型输出只有{self.probs.shape[1]}个类别")
        
        # 计算交叉熵损失
        label_probs = self.probs[np.arange(self.batch_size), labels]
        loss = -np.sum(np.log(label_probs + 1e-10)) / self.batch_size
        
        return loss
    
    def backward(self):
        # 计算梯度
        # 初始化梯度为softmax的输出
        self.grads = self.probs.copy()
        
        # 对于真实标签，减去1
        self.grads[np.arange(self.batch_size), self.labels] -= 1
        
        # 梯度归一化
        self.grads /= self.batch_size
        
        # 将梯度传递给模型进行反向传播
        if self.model:
            self.model.backward(self.grads)
        
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition