import numpy as np
from struct import unpack
import gzip
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

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
        
        # Initialize weights with Xavier initialization
        weight_scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.params['W'] = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) * weight_scale
        self.params['b'] = np.zeros(out_channels)
        
        self.inputs = None
        self.padded = None
        self.output_shape = None
    
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
        self.inputs = inputs
        x_col = self.im2col(self.padded, h_out, w_out)
        
        # Perform convolution as matrix multiplication
        output = (W_col @ x_col.T).T
        output = output.reshape(N, h_out, w_out, self.out_channels).transpose(0, 3, 1, 2)
        
        # Add bias
        output += self.params['b'].reshape(1, self.out_channels, 1, 1)
        
        return output
    
    def col2im(self, cols, x_shape):
        """Transform columns back to images."""
        N, C, H, W = x_shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        # Calculate output dimensions
        H_padded, W_padded = H + 2 * p, W + 2 * p
        h_out = (H_padded - k) // s + 1
        w_out = (W_padded - k) // s + 1
        
        # Initialize padded array
        x_padded = np.zeros((N, C, H_padded, W_padded))
        
        # Reshape cols for processing
        cols_reshaped = cols.reshape(N, h_out, w_out, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        
        # Accumulate values in the padded array
        for h in range(h_out):
            h_start = h * s
            for w in range(w_out):
                w_start = w * s
                x_padded[:, :, h_start:h_start+k, w_start:w_start+k] += cols_reshaped[:, :, :, :, h, w]
        
        # Remove padding if necessary
        if p > 0:
            return x_padded[:, :, p:-p, p:-p]
        return x_padded
    
    def backward(self, grad):
        N, C, H, W = self.inputs.shape
        k = self.kernel_size
        p = self.padding
        s = self.stride
        
        # Calculate output dimensions
        h_out = (H + 2 * p - k) // s + 1
        w_out = (W + 2 * p - k) // s + 1
        
        # Reshape incoming gradient
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Get input columns
        x_col = self.im2col(self.padded, h_out, w_out)
        
        # Calculate gradients for weights and bias
        self.grads['W'] = (grad_reshaped.T @ x_col).reshape(self.params['W'].shape)
        self.grads['b'] = np.sum(grad_reshaped, axis=0)
        
        # Calculate gradient for next layer
        W_col = self.params['W'].reshape(self.out_channels, -1)
        dx_col = grad_reshaped @ W_col
        
        # Convert column gradients back to image format
        dx_padded = self.col2im(dx_col.reshape(N, h_out, w_out, -1), (N, C, H, W))
        
        # Remove padding if applied
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
        
        # Calculate output dimensions
        h_out = (H - pool) // stride + 1
        w_out = (W - pool) // stride + 1
        
        # Initialize output array and mask
        output = np.zeros((N, C, h_out, w_out))
        self.mask = np.zeros_like(inputs)
        
        # Perform max pooling
        for n in range(N):
            for c in range(C):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * stride
                        w_start = w * stride
                        
                        # Select patch
                        patch = inputs[n, c, h_start:h_start+pool, w_start:w_start+pool]
                        
                        # Find maximum value and its position
                        max_val = np.max(patch)
                        max_idx = np.argmax(patch)
                        
                        # Convert 1D index to 2D indices
                        max_h, max_w = np.unravel_index(max_idx, (pool, pool))
                        
                        # Store maximum value in output
                        output[n, c, h, w] = max_val
                        
                        # Save position for backward pass
                        self.mask[n, c, h_start + max_h, w_start + max_w] = 1
        
        self.inputs = inputs
        return output
    
    def backward(self, grad):
        N, C, H_out, W_out = grad.shape
        dx = np.zeros_like(self.inputs)
        
        # Distribute gradient
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        # Distribute gradient to the max position
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
        loss = -np.sum(np.log(self.probs[np.arange(N), labels])) / N
        return loss
    
    def backward(self):
        # Gradient of cross entropy loss
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
        # Update weights using SGD with momentum
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for param in layer.params:
                    # Initialize momentum buffer if not exists
                    if f'v_{param}' not in layer.__dict__:
                        layer.__dict__[f'v_{param}'] = np.zeros_like(layer.params[param])
                    
                    # Update momentum
                    layer.__dict__[f'v_{param}'] = momentum * layer.__dict__[f'v_{param}'] - lr * layer.grads[param]
                    
                    # L2 regularization
                    if param == 'W':  # Only apply weight decay to weights, not biases
                        layer.__dict__[f'v_{param}'] -= lr * weight_decay * layer.params[param]
                    
                    # Update parameters
                    layer.params[param] += layer.__dict__[f'v_{param}']
    
    def predict(self, X):
        outputs = self.forward(X, train=False)
        return np.argmax(outputs, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model on the given data and labels.
        
        Args:
            X: Input data with shape (N, D) for MLP or (N, C, H, W) for CNN
            y: True labels with shape (N,)
            
        Returns:
            accuracy: Classification accuracy
        """
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

def train_mlp():
    """Train multi-layer perceptron model"""
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    if X_train is None:
        return None
    
    # Split training data into training and validation sets
    validation_size = 10000
    X_val = X_train[:validation_size]
    y_val = y_train[:validation_size]
    X_train = X_train[validation_size:]
    y_train = y_train[validation_size:]
    
    # Build neural network
    model = Network()
    model.add(Linear(784, 256))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Linear(256, 128))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Linear(128, 10))
    
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    # Training parameters
    batch_size = 128
    epochs = 6  # Only train for 6 epochs
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # Initialize training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    # Train
    start_time = time.time()
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Initialize loss
        epoch_loss = 0
        correct_predictions = 0
        
        # Train by batches
        for i in tqdm(range(0, len(X_train_shuffled), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
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
            model.update(lr, momentum, weight_decay)
        
        # Calculate average loss and training accuracy
        epoch_loss /= len(X_train_shuffled)
        train_accuracy = correct_predictions / len(X_train_shuffled)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on validation set
        val_accuracy = model.evaluate(X_val, y_val)
        val_accuracies.append(val_accuracy)
        
        # Evaluate on test set
        test_accuracy = model.evaluate(X_test, y_test)
        test_accuracies.append(test_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
        
        # Learning rate schedule
        if epoch > 3:
            lr *= 0.1
        
        # Save model after the 6th epoch
        if epoch == 5:
            save_model(model, 'models/mlp_model_epoch6.pkl')
    
    training_time = time.time() - start_time
    print(f'Training time: {training_time:.2f} seconds')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('MLP Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.plot(test_accuracies, label='Test')
    plt.title('MLP Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/mlp_training_history.png')
    plt.show()
    
    return model

def train_cnn():
    """Train convolutional neural network model"""
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    if X_train is None:
        return None
    
    # Reshape data to (N, C, H, W) format
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Split training data into training and validation sets
    validation_size = 10000
    X_val = X_train[:validation_size]
    y_val = y_train[:validation_size]
    X_train = X_train[validation_size:]
    y_train = y_train[validation_size:]
    
    # Build CNN
    model = Network()
    
    # First convolutional layer, 32 5x5 filters
    model.add(Conv2D(1, 32, 5, padding=2))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Second convolutional layer, 64 3x3 filters
    model.add(Conv2D(32, 64, 3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Linear(64 * 7 * 7, 256))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(256, 10))
    
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    # Training parameters
    batch_size = 64
    epochs = 6  # Only train for 6 epochs
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # Initialize training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    # Train
    start_time = time.time()
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Initialize loss
        epoch_loss = 0
        correct_predictions = 0
        
        # Train by batches
        for i in tqdm(range(0, len(X_train_shuffled), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
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
            model.update(lr, momentum, weight_decay)
        
        # Calculate average loss and training accuracy
        epoch_loss /= len(X_train_shuffled)
        train_accuracy = correct_predictions / len(X_train_shuffled)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on validation set
        val_accuracy = model.evaluate(X_val, y_val)
        val_accuracies.append(val_accuracy)
        
        # Evaluate on test set
        test_accuracy = model.evaluate(X_test, y_test)
        test_accuracies.append(test_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
        
        # Learning rate schedule
        if epoch > 3:
            lr *= 0.1
        
        # Save model after the 6th epoch
        if epoch == 5:
            save_model(model, 'models/cnn_model_epoch6.pkl')
    
    training_time = time.time() - start_time
    print(f'Training time: {training_time:.2f} seconds')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('CNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.plot(test_accuracies, label='Test')
    plt.title('CNN Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/cnn_training_history.png')
    plt.show()
    
    # Visualize CNN filters
    visualize_cnn_filters(model)
    
    return model

def visualize_cnn_filters(model):
    """Visualize filters of the first convolutional layer"""
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    if not conv_layers:
        return
    
    # Get weights of the first convolutional layer
    first_conv = conv_layers[0]
    filters = first_conv.params['W']
    
    # Reshape filters for visualization
    n_filters = filters.shape[0]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot each filter
    for i in range(min(32, n_filters)):
        plt.subplot(4, 8, i + 1)
        plt.imshow(filters[i, 0], cmap='gray')
        plt.axis('off')
    
    plt.suptitle('First Layer CNN Filters')
    plt.tight_layout()
    plt.savefig('plots/cnn_filters.png')
    plt.show()

def data_augmentation(X, y, num_augmentations=1):
    """Apply data augmentation to generate more samples"""
    N, C, H, W = X.shape
    X_aug = []
    y_aug = []
    
    for i in range(N):
        # Original image
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Augmented images
        for _ in range(num_augmentations):
            img = X[i, 0]  # Get image (remove channel dimension)
            
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
            
            # Add augmented image
            X_aug.append(img_shifted.reshape(1, H, W))
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

def experiment_different_architectures():
    """Test different network architectures and hyperparameters"""
    # Load data
    X_train_full, y_train_full, X_test, y_test = load_mnist()
    if X_train_full is None:
        return None
    
    # Reshape data for CNN
    X_train_cnn_full = X_train_full.reshape(-1, 1, 28, 28)
    X_test_cnn = X_test.reshape(-1, 1, 28, 28)
    
    # Split training data
    validation_size = 10000
    X_val = X_train_full[:validation_size]
    y_val = y_train_full[:validation_size]
    X_train = X_train_full[validation_size:]
    y_train = y_train_full[validation_size:]
    
    # Same for CNN data
    X_val_cnn = X_train_cnn_full[:validation_size]
    X_train_cnn = X_train_cnn_full[validation_size:]
    
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    
    # Define different architectures to test
    architectures = [
        {
            'name': 'Simple MLP',
            'layers': [
                Linear(784, 100),
                ReLU(),
                Linear(100, 10)
            ]
        },
        {
            'name': 'Deep MLP',
            'layers': [
                Linear(784, 256),
                ReLU(),
                Dropout(0.3),
                Linear(256, 128),
                ReLU(),
                Dropout(0.3),
                Linear(128, 64),
                ReLU(),
                Linear(64, 10)
            ]
        },
        {
            'name': 'Simple CNN',
            'is_cnn': True,
            'layers': [
                Conv2D(1, 16, 5, padding=2),
                ReLU(),
                MaxPool2D(2),
                Flatten(),
                Linear(16 * 14 * 14, 100),
                ReLU(),
                Linear(100, 10)
            ]
        },
        {
            'name': 'Deep CNN',
            'is_cnn': True,
            'layers': [
                Conv2D(1, 32, 5, padding=2),
                ReLU(),
                MaxPool2D(2),
                Conv2D(32, 64, 3, padding=1),
                ReLU(),
                MaxPool2D(2),
                Flatten(),
                Linear(64 * 7 * 7, 256),
                ReLU(),
                Dropout(0.5),
                Linear(256, 10)
            ]
        }
    ]
    
    # Hyperparameters
    batch_size = 128
    epochs = 6  # Train for 6 epochs
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # Results
    results = []
    
    # Test each architecture
    for idx, arch in enumerate(architectures):
        print(f"\nTesting {arch['name']}...")
        
        # Create model with specified architecture
        model = Network()
        for layer in arch['layers']:
            model.add(layer)
        
        loss_function = CrossEntropyLoss()
        model.set_loss(loss_function)
        
        # Choose appropriate data
        if arch.get('is_cnn', False):
            X_train_arch = X_train_cnn
            X_val_arch = X_val_cnn
            X_test_arch = X_test_cnn
        else:
            X_train_arch = X_train
            X_val_arch = X_val
            X_test_arch = X_test
        
        # Train
        start_time = time.time()
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train_arch))
            X_train_shuffled = X_train_arch[indices]
            y_train_shuffled = y_train[indices]  # Make sure this is y_train, not y_train_full
            
            # Initialize loss
            epoch_loss = 0
            correct_predictions = 0
            
            # Train by batches
            for i in tqdm(range(0, len(X_train_shuffled), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
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
                model.update(lr, momentum, weight_decay)
            
            # Calculate average loss and training accuracy
            epoch_loss /= len(X_train_shuffled)
            train_accuracy = correct_predictions / len(X_train_shuffled)
            train_losses.append(epoch_loss)
            train_accuracies.append(train_accuracy)
            
            # Evaluate on validation set
            val_accuracy = model.evaluate(X_val_arch, y_val)
            val_accuracies.append(val_accuracy)
            
            # Evaluate on test set
            test_accuracy = model.evaluate(X_test_arch, y_test)
            test_accuracies.append(test_accuracy)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
            
            # Save model after the 6th epoch
            if epoch == 5:
                save_model(model, f'models/arch_{idx+1}_{arch["name"]}_epoch6.pkl')
        
        training_time = time.time() - start_time
        
        # Save results
        results.append({
            'name': arch['name'],
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_accuracies': test_accuracies,
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1],
            'final_test_accuracy': test_accuracies[-1],
            'training_time': training_time
        })
        
        print(f'Final test accuracy: {test_accuracies[-1]:.4f}')
        print(f'Training time: {training_time:.2f} seconds')
        
        # Plot training history for this architecture
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'{arch["name"]} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training')
        plt.plot(val_accuracies, label='Validation')
        plt.plot(test_accuracies, label='Test')
        plt.title(f'{arch["name"]} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/arch_{idx+1}_{arch["name"]}_history.png')
        plt.show()
    
    # Print comparison results
    print("\nArchitecture Comparison Results:")
    print("-" * 80)
    print(f"{'Architecture':<20} {'Train Acc':<15} {'Val Acc':<15} {'Test Acc':<15} {'Training Time':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} {result['final_train_accuracy']:.4f}{'':<9} {result['final_val_accuracy']:.4f}{'':<9} {result['final_test_accuracy']:.4f}{'':<9} {result['training_time']:.2f}s")
    
    # Plot validation accuracy comparison
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(range(1, epochs + 1), result['val_accuracies'], marker='o', label=result['name'])
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/architecture_comparison.png')
    plt.show()
    
    return results

def train_cnn_with_augmentation():
    """Train CNN with data augmentation"""
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    if X_train is None:
        return None
    
    # Reshape data to (N, C, H, W) format
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Split training data into training and validation sets
    validation_size = 10000
    X_val = X_train[:validation_size]
    y_val = y_train[:validation_size]
    X_train = X_train[validation_size:]
    y_train = y_train[validation_size:]
    
    # Apply data augmentation to training set
    print("Applying data augmentation...")
    X_train_aug, y_train_aug = data_augmentation(X_train, y_train, num_augmentations=1)
    print(f"Original training set size: {len(X_train)}")
    print(f"Augmented training set size: {len(X_train_aug)}")
    
    # Build CNN
    model = Network()
    
    # First convolutional layer, 32 5x5 filters
    model.add(Conv2D(1, 32, 5, padding=2))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Second convolutional layer, 64 3x3 filters
    model.add(Conv2D(32, 64, 3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(2))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Linear(64 * 7 * 7, 256))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(256, 10))
    
    loss_function = CrossEntropyLoss()
    model.set_loss(loss_function)
    
    # Training parameters
    batch_size = 64
    epochs = 6  # Only train for 6 epochs
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # Initialize training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    # Train
    start_time = time.time()
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train_aug))
        X_train_shuffled = X_train_aug[indices]
        y_train_shuffled = y_train_aug[indices]
        
        # Initialize loss
        epoch_loss = 0
        correct_predictions = 0
        
        # Train by batches
        for i in tqdm(range(0, len(X_train_shuffled), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
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
            model.update(lr, momentum, weight_decay)
        
        # Calculate average loss and training accuracy
        epoch_loss /= len(X_train_shuffled)
        train_accuracy = correct_predictions / len(X_train_shuffled)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on validation set
        val_accuracy = model.evaluate(X_val, y_val)
        val_accuracies.append(val_accuracy)
        
        # Evaluate on test set
        test_accuracy = model.evaluate(X_test, y_test)
        test_accuracies.append(test_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
        
        # Learning rate schedule
        if epoch > 3:
            lr *= 0.1
        
        # Save model after the 6th epoch
        if epoch == 5:
            save_model(model, 'models/cnn_aug_model_epoch6.pkl')
    
    training_time = time.time() - start_time
    print(f'Training time: {training_time:.2f} seconds')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('CNN+Aug Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.plot(test_accuracies, label='Test')
    plt.title('CNN+Aug Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/cnn_aug_training_history.png')
    plt.show()
    
    # Visualize CNN filters
    visualize_cnn_filters(model)
    
    return model

def main():
    """Main function"""
    # Create model and plot directories
    create_model_directory()
    
    print("MNIST Neural Network Training")
    print("=" * 40)
    print("Training options:")
    print("1. Multi-layer Perceptron (MLP)")
    print("2. Convolutional Neural Network (CNN)")
    print("3. Convolutional Neural Network (CNN) with data augmentation")
    print("4. Test different architectures")
    print("5. Exit")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '1':
        print("\nTraining MLP...")
        model = train_mlp()
    
    elif choice == '2':
        print("\nTraining CNN...")
        model = train_cnn()
    
    elif choice == '3':
        print("\nTraining CNN with data augmentation...")
        model = train_cnn_with_augmentation()
    
    elif choice == '4':
        print("\nTesting different architectures...")
        results = experiment_different_architectures()
        # Find the best model
        if results:
            best_result = max(results, key=lambda x: x['final_test_accuracy'])
            print(f"\nBest model: {best_result['name']} with test accuracy of {best_result['final_test_accuracy']:.4f}")
    
    elif choice == '5':
        print("\nExiting...")
        return
    
    else:
        print("\nInvalid choice. Exiting.")
        return
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()