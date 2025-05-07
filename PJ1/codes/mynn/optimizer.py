from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model
        self.debug = True  # 添加调试标志，首次运行打印调试信息

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    # 确保梯度存在
                    if layer.grads[key] is not None:
                        # 记录更新前的参数
                        old_param = layer.params[key].copy()
                        
                        # 权重衰减
                        if layer.weight_decay:
                            layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                        
                        # 梯度更新
                        layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
                        
                        # 确保类属性与params字典一致
                        if hasattr(layer, key):
                            setattr(layer, key, layer.params[key])

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        """
        带动量的梯度下降优化器
        init_lr: 初始学习率
        model: 要优化的模型
        mu: 动量系数，默认为0.9
        """
        super().__init__(init_lr, model)
        self.mu = mu
        
        # 初始化每个参数的动量为0
        self.velocity = {}
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocity[i] = {}
                for key in layer.params.keys():
                    self.velocity[i][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        """
        执行一步优化
        使用公式: v = mu * v - lr * gradient
                 w = w + v
        """
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.grads[key] is not None:  # 确保梯度存在
                        # 如果使用权重衰减(L2正则化)
                        if layer.weight_decay:
                            # 等价于在梯度中添加lambda*W，但这里直接在参数上应用衰减
                            layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                        
                        # 更新速度(动量)
                        self.velocity[i][key] = self.mu * self.velocity[i][key] - self.init_lr * layer.grads[key]
                        
                        # 更新参数
                        layer.params[key] += self.velocity[i][key]