"""
We implement the main 2-layer CNN network here
"""
import numpy as np
np.random.seed(42)

# Activation functions
def relu(x):
    """
    ReLU activation
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax activation
    x: (B, C)
    """
    exponentials = np.exp(x) # (B, C)
    exponentials_sum = np.sum(exponentials, axis=1, keepdims=True) # (B, )
    softmax_outputs = exponentials / (exponentials_sum + 1e-8) # (B, C) probabilities
    return softmax_outputs


class ConvLayer:
    """
    A Simple Conv layer using Numpy only
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        std = np.sqrt(2.0/(in_channels*kernel_size*kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)/std # Trainable
        self.biases = np.zeros(out_channels) # Trainable
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Forward pass
        x.shape = (B, C, H, W) like [32, 1, 28, 28]
        weights.shape = (c_out, c_in, kernel_size, kernel_size) [16, 1, 3, 3]
        After im2col:
        patches.shape = (B, h_out*w_out, kernel_size*kernel_size*channels) [32, 784, 9]
        """
        # Work with channels first
        image_patches, height_out, width_out = self._im2col(x)
        # Reshape the weights
        c_out, _, _, _ = self.weights.shape
        weights_reshaped = np.reshape(self.weights, (c_out, -1)) # (16, 9)
        outputs = image_patches @ weights_reshaped.T # (32, 784, 9) @ (9, 16) --> (32, 784, 16)
        outputs = outputs + self.biases # (32, 784, 16) + (16,) --> (32, 784, 16) via broadcasting
        # Final reshape to [B, c_out, h_out, w_out]
        outputs = np.reshape(outputs, (image_patches.shape[0],height_out, width_out, c_out))
        outputs = outputs.transpose(0, 3, 1, 2)
        return outputs
    
    def _im2col(self, image):
        """
        Extract patches
        """
        batch_size, channels, height, width = image.shape
        # Pad image, ONLY ALONG the height and width dimensions
        # Keep the batch size and channels as they are!
        if self.padding > 0:
            padded_image = np.pad(
                image,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            padded_image = image
        # Calculate the output dimensions
        height_out = (height + 2*self.padding-self.kernel_size) // self.stride + 1
        width_out = (width + 2*self.padding-self.kernel_size) // self.stride + 1
        # Get the strides
        s0, s1, s2, s3 = padded_image.strides
        # Create the strided view
        patches = np.lib.stride_tricks.as_strided(
            padded_image,
            shape=(batch_size, height_out, width_out, self.kernel_size, self.kernel_size, channels),
            strides=(s0, self.stride*s2, self.stride*s3, s2, s3, s1)
        )
        # Reshape to (num_patches, self.kernel_size*self.kernel_size*channels)
        patches = np.reshape(patches, (batch_size, height_out*width_out, self.kernel_size*self.kernel_size*channels))
        return patches, height_out, width_out

class MaxPool:
    """
    Max-pooling layer
    """
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x):
        """
        Forward pass through the max-pooling layer
        (batch, channels, height_in, width_in) --> (batch, channels, height_in/pool_size, width_in//pool_size)
        (32, 16, 28, 28) --> (32, 16, 14, 14)
        """
        B, C, H, W = x.shape
        
        # Output dimensions
        h_out = H//self.pool_size
        w_out = W//self.pool_size

        # Reshape to create the pooling windows
        # (B, C, H_out, pool_size, W_out, pool_size)
        x_reshaped = x.reshape(B, C, h_out, self.pool_size, w_out, self.pool_size) # (32, 16, 14, 2, 14, 2)

        # Max-pooling over dimensions 3 and 5
        x_reshaped = np.max(x_reshaped, axis=3) # Dimension 3
        x_reshaped = np.max(x_reshaped, axis=4) # Dimension 5

        assert x_reshaped.shape == (B, C, h_out, w_out)
        return x_reshaped

class FCLayer:
    """
    Fully-connected layer
    """
    def __init__(self, in_channels, out_channels):
        std = np.sqrt(2.0/in_channels)
        self.weights = np.random.randn(out_channels, in_channels)*std # PyTorch convention is to put the output channels first
        self.biases = np.zeros(out_channels)
    
    def forward(self, x):
        """
        Forward pass
        Feature map usually has 4 dimensions
        Weights and biases have 2 and 1 respectively
        """
        if len(x.shape) > 2:
            B, C, H, W = x.shape
            x = np.reshape(x, (B, C*H*W))
        weights_reshaped = self.weights.T # (out, C*H*W) --> (C*H*W, out)
        outputs = x @ weights_reshaped # (B, C*H*W) @ (C*H*W, out_channels) --> (B, out_channels)
        outputs = outputs + self.biases # (B, out_channels) + (out_channels,) --> (B, out_channels)
        return outputs


class SimpleCNN:
    """
    CNN using only numpy
    """
    def __init__(self):
        """
        Architecture:
        - Conv: 1→16 filters, 3x3
        - ReLU
        - MaxPool: 2x2
        - Flatten
        - FC: 16*14*14 → 128
        - ReLU
        - FC: 128 → 10
        - Softmax
        """
        self.conv1 = ConvLayer(1, 16, 3, stride=1, padding=1)
        self.max_pool = MaxPool(2, 2)
        self.fc1 = FCLayer(16*14*14, 128)
        self.fc2 = FCLayer(128, 10)
    
    def forward(self, X):
        """
        Complete forward pass
        # Conv → ReLU → Pool
        # Flatten
        # FC → ReLU → FC
        # Softmax
        """
        if len(X.shape) == 3:
            B, H, W = X.shape
            X = np.reshape(X, (B, 1, H, W))
        elif X.shape[3] == 1:
            B, H, W, 1 = X.shape
            X = np.reshape(X, (B, 1, H, W))
        # Convolution --> ReLU --> Pooling
        out = self.conv1.forward(X)
        out = relu(out)
        out = self.max_pool.forward(X)

        # FC --> ReLU
        out = self.fc1.forward(out)
        out = relu(out)

        # FC --> Softmax
        logits = self.fc2.forward(out)
        probs = softmax(logits)

        return probs
