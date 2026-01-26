"""
Functions to do backprop algorithm and update the gradients for each param
dL/dW_layer = dL/dout × dout/dW_layer
            = gradient_from_next × local_gradient
"""
import numpy as np
from utils import im2col

def fc_backward(d_out, x, weights):
    """
    d_out: Gradient from the next layer (B, out_features)
    B: Batch size
    x: input used in forward (B, in_features)
    weights: (out_features, in_features)
    Returns the gradients of dX, dW and dB
    d_out = x @ weights.transpose + biases --> Forward pass
    (B, in) @ (in, out) --> (B, out)
    """
    # Gradient wrt the input
    dx = d_out @ weights # (B, in)
    # Gradient wrt the weights
    dw = d_out.T @ x # (out, B) @ (B, in) --> (out, in)
    # Gradient wrt the bias
    db = np.sum(d_out, axis=0) # (out, )
    return dx, dw, db

def relu_backward(d_out, x):
    """
    Derivative of the ReLU output
    """
    dx = d_out * (x > 0) # For vectorized values, use a mask
    return dx

def maxpool_backward(d_out, x, pool_size=2):
    """
    d_out: Gradient from the next layer, (B, C, h_out, w_out)
    x: Input from the forward pass (B, C, H, W)
    pool_size: Pooling window
    Returns: dx (B, C, H, W)
    """
    B, C, H, W = x.shape
    h_out = H//pool_size
    w_out = W//pool_size
    dx = np.zeros_like(x) # (B, C, H, W)
    for b in range(B):
        for c in range(C):
            for i in range(h_out):
                for j in range(w_out):
                    # Extract window
                    h_start = i*pool_size
                    w_start = j*pool_size
                    window = x[
                        b, c, h_start:h_start+pool_size,
                        w_start:w_start+pool_size
                    ]
                    # Find the maximum position in window
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    # Route the gradient to that position
                    dx[b, c, h_start+max_idx[0], w_start+max_idx[1]] += d_out[b, c, i, j]
    return dx

def conv_backward(d_out, x, weights, stride=1, padding=0):
    """
    dout: gradient from next layer (B, out_channels, H_out, W_out)
    x: input from forward (B, in_channels, H, W)
    weights: (out_channels, in_channels, kH, kW)
    Returns: dx, dW, db
    """
    B, C_in, H, W = x.shape
    c_out, _, kernel_height, kernel_width = weights.shape

    # Step 1: Gradient wrt the bias
    # db: Sum gradient over all spatial positions and batch
    # d_out: (B, c_out, h_out, w_out) --> db: (c_out,)
    db = np.sum(d_out, axis=(0, 2, 3)) # Sum over B, h_out, w_out

    # Step 2: Gradient wrt the weights
    # Need patches.T @ d_out
    # Gives dW in flattened form
    # pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    # extract patches
    patches, _, _ = im2col(x_padded, padding, kernel_height, stride) # Assume kernel_height == kernel_width
    # Reshape d_out
    d_out_reshaped = d_out.transpose(0, 2, 3, 1)  # (B, H_out, W_out, C_out)
    d_out_reshaped = d_out_reshaped.reshape(B, -1, c_out)
    # Compute dW
    dw = np.zeros_like(weights)
    for b in range(B):
        dw_flat = patches[b].T @ d_out_reshaped[b] # (c_in*kernel_height*kernel_width, c_out)
        dw += dw_flat.T.reshape(c_out, C_in, kernel_height, kernel_width)
    dw /= B

    # Step 3: Gradient wrt the input
    # Forward pass was conv(x, W), hence backward is conv(d_out, W_flipped)
    # Flip weights
    w_flipped = np.flip(weights, axis=(2, 3))
    # Pad d_out for full convolution
    pad_h = kernel_height - 1
    pad_w = kernel_width - 1
    d_out_padded = np.pad(d_out, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)))
    # Convolve d_out with flipped weights
    dx = np.zeros_like(x)
    for b in range(B):
        for c_in in range(C_in):
            for c_o in range(c_out):
                for h in range(H):
                    for w in range(W):
                        h_start, w_start = h, w
                        region = d_out_padded[b, c_o, h_start:h_start+kernel_height, w_start:w_start+kernel_width]
                        # Accumulate the gradients
                        dx[b, c_in, h, w] += np.sum(
                            region*w_flipped[c_o, c_in]
                        )
    return dx, dw, db
