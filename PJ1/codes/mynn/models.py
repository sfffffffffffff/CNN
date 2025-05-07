from .op import *
import pickle
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.input = X  # 保存输入形状，用于反向传播
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)  # 将输入展平为[batch_size, features]
    
    def backward(self, grad):
        return grad.reshape(self.input.shape)  # 恢复原始形状
class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        
class Model_CNN(Layer):
    """
    A model with conv2D layers. Optimized implementation to improve speed and consistency.
    """
    def __init__(self, conv_params, fc_size_list, act_func='ReLU', lambda_list=None):
        super().__init__()
        self.conv_params = conv_params
        self.fc_size_list = fc_size_list
        self.act_func = act_func
        
        if conv_params is not None and fc_size_list is not None:
            self.layers = []
            
            # Add convolutional layers
            for i, params in enumerate(conv_params):
                # Extract parameters with default values
                in_channels = params['in_channels']
                out_channels = params['out_channels']
                kernel_size = params['kernel_size']
                stride = params.get('stride', 1)
                padding = params.get('padding', 0)
                
                # Create convolutional layer
                conv_layer = conv2D(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
                
                if lambda_list is not None and i < len(lambda_list):
                    conv_layer.weight_decay = True
                    conv_layer.weight_decay_lambda = lambda_list[i]
                
                self.layers.append(conv_layer)
                
                # Add activation function
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError("Logistic activation not implemented")
            
            # Add a Flatten layer
            self.layers.append(Flatten())
            
            # Add fully connected layers
            for i in range(len(fc_size_list) - 1):
                fc_layer = Linear(in_dim=fc_size_list[i], out_dim=fc_size_list[i + 1])
                
                if lambda_list is not None and i + len(conv_params) < len(lambda_list):
                    fc_layer.weight_decay = True
                    fc_layer.weight_decay_lambda = lambda_list[i + len(conv_params)]
                
                self.layers.append(fc_layer)
                
                # Add activation function except for the last layer
                if i < len(fc_size_list) - 2:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    elif act_func == 'Logistic':
                        raise NotImplementedError("Logistic activation not implemented")

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch_size, channels, height, width]
        Forward pass through the network in a more efficient manner.
        """
        outputs = X
        
        # Forward pass through all layers
        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs

    def backward(self, loss_grad):
        """
        Backward pass through the network.
        """
        grads = loss_grad
        
        # Backward pass through all layers in reverse order
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        
        return grads
    
    def load_model(self, param_path):
        """
        Load model parameters from a file.
        """
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        
        self.conv_params = param_list[0]
        self.fc_size_list = param_list[1]
        self.act_func = param_list[2]
        
        # Rebuild network structure
        self.layers = []
        layer_idx = 3  # Parameter list starting index
        
        # Add convolutional layers
        for i, params in enumerate(self.conv_params):
            in_channels = params['in_channels']
            out_channels = params['out_channels']
            kernel_size = params['kernel_size']
            stride = params.get('stride', 1)
            padding = params.get('padding', 0)
            
            # Create convolutional layer
            conv_layer = conv2D(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
            
            # Load weights and biases
            conv_layer.W = param_list[layer_idx]['W']
            conv_layer.b = param_list[layer_idx]['b']
            conv_layer.params['W'] = conv_layer.W
            conv_layer.params['b'] = conv_layer.b
            conv_layer.weight_decay = param_list[layer_idx]['weight_decay']
            conv_layer.weight_decay_lambda = param_list[layer_idx]['lambda']
            layer_idx += 1
            
            self.layers.append(conv_layer)
            
            # Add activation function
            if self.act_func == 'ReLU':
                self.layers.append(ReLU())
            elif self.act_func == 'Logistic':
                raise NotImplementedError("Logistic activation not implemented")
        
        # Add Flatten layer
        self.layers.append(Flatten())
        
        # Add fully connected layers
        for i in range(len(self.fc_size_list) - 1):
            fc_layer = Linear(in_dim=self.fc_size_list[i], out_dim=self.fc_size_list[i + 1])
            
            # Load weights and biases
            fc_layer.W = param_list[layer_idx]['W']
            fc_layer.b = param_list[layer_idx]['b']
            fc_layer.params['W'] = fc_layer.W
            fc_layer.params['b'] = fc_layer.b
            fc_layer.weight_decay = param_list[layer_idx]['weight_decay']
            fc_layer.weight_decay_lambda = param_list[layer_idx]['lambda']
            layer_idx += 1
            
            self.layers.append(fc_layer)
            
            # Add activation function except for the last layer
            if i < len(self.fc_size_list) - 2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'Logistic':
                    raise NotImplementedError("Logistic activation not implemented")
        
    def save_model(self, save_path):
        """
        Save model parameters to a file.
        """
        param_list = [self.conv_params, self.fc_size_list, self.act_func]
        
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'], 
                    'b': layer.params['b'], 
                    'weight_decay': layer.weight_decay, 
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)