"""
Keep a set of frequently used functions here
"""
import numpy as np

def relu(x):
    """
    Rectified Linear Unit
    relu(x) = max(0, x)
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Implement the Softmax activation for an array
    [B, C]
    """
    exponentials = np.exp(x)
    exponentials_sum = np.sum(exponentials, axis=1, keepdims=True)
    softmax_outputs = exponentials/(exponentials_sum + 1e-8)
    return softmax_outputs # (B, C)

def im2col(image, padding, kernel_size, stride):
    """
    Extract patches from an image
    """
    batch_size, channels, height, width = image.shape
    # Pad image, only along the height and width dimensions
    # Done BEFORE the patches are extracted
    if padding > 0:
        padded_image = np.pad(
            image,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        padded_image = image
    # Determine the output dimensions
    height_out = (height + 2*padding-kernel_size)//stride + 1
    width_out = (width + 2*padding-kernel_size)//stride + 1
    # Get the current strides
    s0, s1, s2, s3 = padded_image.strides # Corresponds to batch, channels, height and width
    # Create the strided view
    patches = np.lib.stride_tricks.as_strided(
        padded_image,
        shape=(batch_size, height_out, width_out, kernel_size, kernel_size, channels),
        strides=(s0, s2*stride, s3*stride, s2, s3, s1)
    )
    # Reshape to (num_patches, kernel_size*kernel_size*channels)
    patches = np.reshape(patches, (batch_size, height_out*width_out, kernel_size*kernel_size*channels))
    return patches, height_out, width_out

def categorical_cross_entropy(probs, labels):
    """
    Categorical Cross-entropy loss
    loss = -sum(P(x)*log(Q(x)))/batch_size
    P(x): True labels = [[0, 0, 1], [1, 0, 0], [0, 1, 0]], shape [B, C], one-hot encoded
    Q(x): Probabilities = [[0.1, 0.1, 0.8], [0.1, 0.5, 0.4], [0.6, 0.2, 0.2]] shape [B, C]
    Here, the batch size is 3
    loss = -(1/3) * {(0*log(0.1) + 0*log(0.1) + 1*log(0.8)) + (1*log(0.1) + 0*log(0.5) + 0*log(0.4)) + (0*log(0.6) + 1*log(0.2) + 0*log(0.2))}
    """
    # Obtain the dot products
    losses = labels * np.log(probs) # P(x) * log{Q(x)}
    losses = np.sum(losses, axis=1) # Sum the values across each column, FOR EACH sample in the batch
    loss = -np.mean(losses)
    return loss.item()
