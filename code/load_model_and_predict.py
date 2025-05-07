import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
from struct import unpack
import os
from tqdm import tqdm
import sys

# 确保当前目录在系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 方法1：在同一个文件中重新定义所有需要的类
# 这确保了pickle可以在加载时找到这些类

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
        loss = -np.sum(np.log(self.probs[np.arange(N), labels])) / N
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
        
    def add(self, layer):
        self.layers.append(layer)
        if isinstance(layer, Dropout):
            self.dropout_layers.append(layer)
    
    def set_loss(self, loss_function):
        self.loss_function = loss_function
    
    def forward(self, X, train=True):
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                outputs = layer.forward(outputs, train)
            else:
                outputs = layer.forward(outputs)
        return outputs
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self, lr, momentum=0.9, weight_decay=1e-4):
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for param in layer.params:
                    if f'v_{param}' not in layer.__dict__:
                        layer.__dict__[f'v_{param}'] = np.zeros_like(layer.params[param])
                    
                    layer.__dict__[f'v_{param}'] = momentum * layer.__dict__[f'v_{param}'] - lr * layer.grads[param]
                    
                    if param == 'W':
                        layer.__dict__[f'v_{param}'] -= lr * weight_decay * layer.params[param]
                    
                    layer.params[param] += layer.__dict__[f'v_{param}']
    
    def predict(self, X):
        outputs = self.forward(X, train=False)
        return np.argmax(outputs, axis=1)
    
    def evaluate(self, X, y):
        assert X.shape[0] == y.shape[0], f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}"
        
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

def set_random_seed(seed=42):
    """Set random seed to ensure reproducible results"""
    np.random.seed(seed)

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
    
    # Normalize data, ensuring same preprocessing as during training
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    return X_train, y_train, X_test, y_test

# 方法2：自定义更健壮的模型加载函数
def load_model(filename):
    """Load complete model using pickle with custom handling"""
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
    except ModuleNotFoundError as e:
        print(f"Module not found error: {str(e)}")
        print("This error typically happens when pickle can't find the classes used in the model.")
        print("Make sure that all required class definitions are in this script.")
        return None
    except AttributeError as e:
        print(f"Attribute error: {str(e)}")
        print("This error happens when the loaded model contains attributes or classes that can't be found.")
        print("Make sure that all required class definitions are in this script with exactly the same names.")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_single_image(model, image, label=None):
    """Predict a single image and display the result"""
    # Ensure image is the correct shape
    if len(image.shape) == 1:
        # For MLP, keep as flattened vector
        X = image.reshape(1, -1)
    else:
        # For CNN, ensure it's (N, C, H, W) format
        X = image.reshape(1, 1, 28, 28)
    
    # Predict
    prediction = model.predict(X)[0]
    
    # Display image and prediction
    plt.figure(figsize=(5, 5))
    if len(image.shape) == 1:
        plt.imshow(image.reshape(28, 28), cmap='gray')
    else:
        plt.imshow(image[0], cmap='gray')
    
    title = f"Prediction: {prediction}"
    if label is not None:
        title += f" (Actual: {label})"
        if prediction == label:
            title += " ✓"
        else:
            title += " ✗"
    
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    return prediction

def predict_random_samples(model, X_test, y_test, n=5, is_cnn=False):
    """Predict random samples and display results"""
    # Randomly select n samples
    indices = np.random.choice(len(X_test), n, replace=False)
    
    for idx in indices:
        image = X_test[idx]
        label = y_test[idx]
        
        # Adjust image format based on model type
        if is_cnn and len(image.shape) == 1:
            image = image.reshape(1, 28, 28)
        
        predict_single_image(model, image, label)

def evaluate_model(model, X_test, y_test, is_cnn=False):
    """Evaluate model and calculate accuracy"""
    # Adjust image format based on model type
    if is_cnn and len(X_test.shape) == 2:
        X_test_reshaped = X_test.reshape(-1, 1, 28, 28)
    else:
        X_test_reshaped = X_test
    
    # Evaluate in batches to avoid memory issues
    batch_size = 1000
    n_samples = len(X_test)
    n_correct = 0
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Evaluating model"):
        end = min(i + batch_size, n_samples)
        X_batch = X_test_reshaped[i:end]
        y_batch = y_test[i:end]
        
        # Ensure correct prediction mode is used
        predictions = model.predict(X_batch)
        n_correct += np.sum(predictions == y_batch)
    
    accuracy = n_correct / n_samples
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy

def list_available_models():
    """List available model files"""
    if not os.path.exists('models'):
        print("Error: Model directory not found")
        return []
    
    # 允许加载.pkl和.json文件（以防万一）
    model_files = [f for f in os.listdir('models') if f.endswith(('.pkl', '.json'))]
    return model_files

def main():
    """Main function"""
    # Set random seed to ensure reproducible results
    set_random_seed(42)
    
    print("MNIST Neural Network Prediction")
    print("=" * 40)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    if X_test is None:
        print("Could not load dataset. Exiting.")
        return
    
    # List available models
    model_files = list_available_models()
    if not model_files:
        print("No model files found. Please train models first.")
        return
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    # Select model
    try:
        choice = int(input("\nPlease select model number: "))
        if choice < 1 or choice > len(model_files):
            print("Invalid choice. Exiting.")
            return
        
        selected_model_file = model_files[choice-1]
    except ValueError:
        print("Please enter a valid number. Exiting.")
        return
    
    # Determine model type (MLP or CNN)
    is_cnn = 'cnn' in selected_model_file.lower()
    
    # Load selected model
    print(f"\nLoading model '{selected_model_file}'...")
    model = load_model(os.path.join('models', selected_model_file))
    if model is None:
        return
    
    # Prediction options
    print("\nPrediction options:")
    print("1. Evaluate model performance on test set")
    print("2. Predict random samples")
    print("3. Interactive prediction for individual images")
    print("4. Exit")
    
    predict_choice = input("\nPlease enter your choice (1-4): ")
    
    if predict_choice == '1':
        # Evaluate model
        if is_cnn:
            X_test_reshaped = X_test.reshape(-1, 1, 28, 28)
            evaluate_model(model, X_test_reshaped, y_test)
        else:
            evaluate_model(model, X_test, y_test)
    
    elif predict_choice == '2':
        # Predict random samples
        n_samples = int(input("How many random samples to predict? "))
        predict_random_samples(model, X_test, y_test, n=n_samples, is_cnn=is_cnn)
    
    elif predict_choice == '3':
        # Interactive prediction
        while True:
            try:
                idx = int(input("\nEnter index of image in test set (0-9999), or -1 to exit: "))
                if idx == -1:
                    break
                if idx < 0 or idx >= len(X_test):
                    print(f"Index must be between 0 and {len(X_test)-1}")
                    continue
                
                image = X_test[idx]
                label = y_test[idx]
                
                if is_cnn:
                    image = image.reshape(1, 28, 28)
                
                predict_single_image(model, image, label)
            except ValueError:
                print("Please enter a valid number")
    
    elif predict_choice == '4':
        print("\nExiting...")
        return
    
    else:
        print("\nInvalid choice. Exiting.")
        return
    
    print("\nPrediction completed!")

if __name__ == "__main__":
    main()