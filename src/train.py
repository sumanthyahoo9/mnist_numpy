"""
Training script
"""
import numpy as np
from network import SimpleCNN
from load_mnist_data import get_data
from utils import categorical_cross_entropy, relu, softmax
from backpropagation import conv_backward, fc_backward, relu_backward, maxpool_backward

def train(model, X_train, y_train, epochs=10, batch_size=32, learning_rate=0.01):
    """
    The full training loop
    """
    train_losses = []
    N = len(X_train)
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        epoch_loss, num_batches = 0, 0
        for i in range(0, N - batch_size+1, batch_size):
            images = X_shuffled[i:i+batch_size]
            labels = y_shuffled[i:i+batch_size]
            probs = model.forward(images)
            # Forward pass and we save intermediaries
            x_conv = images
            z_conv = model.conv1.forward(x_conv)
            a_conv = relu(z_conv)

            x_pool = a_conv
            z_pool = model.max_pool.forward(x_pool)

            x_fc1 = z_pool
            z_fc1 = model.fc1.forward(x_fc1)
            a_fc1 = relu(z_fc1)

            x_fc2 = a_fc1
            logits = model.fc2.forward(x_fc2)
            probs = softmax(logits)
            loss, grads = categorical_cross_entropy(probs, labels)
            # Backward pass
            dLogits = grads
            # FC2 backward
            dx_fc2, dW_fc2, db_fc2 = fc_backward(dLogits, x_fc2, model.fc2.weights)
            model.fc2.weights -= learning_rate*dW_fc2
            model.fc2.biases -= learning_rate*db_fc2
            # ReLU backward
            da_fc1 = relu_backward(dx_fc2, z_fc1)
            # FC1 backward
            dx_fc1, dW_fc1, db_fc1 = fc_backward(da_fc1, x_fc1, model.fc1.weights)
            model.fc1.weights -= learning_rate * dW_fc1
            model.fc1.biases -= learning_rate * db_fc1
            # Pool backward
            dz_pool = maxpool_backward(dx_fc1, x_pool, model.max_pool.pool_size)
            # ReLU backward
            da_conv = relu_backward(dz_pool, z_conv)
            
            # Conv backward
            _, dW_conv, db_conv = conv_backward(
                da_conv, x_conv, model.conv1.weights,
                model.conv1.stride, model.conv1.padding
            )
            model.conv1.weights -= learning_rate * dW_conv
            model.conv1.biases -= learning_rate * db_conv
            # Update loss values
            epoch_loss += loss
            num_batches += 1
        # Epoch metrics
        avg_loss = epoch_loss/num_batches
        train_losses.append(avg_loss)
    return epochs, train_losses

if __name__ == "__main__":
    model = SimpleCNN()
    X_train, y_train, _, _ = get_data()
    epochs, loss_curve_values = train(model, X_train=X_train, y_train=y_train)
    print(f"After training for {epochs} epochs, the minimum training loss is {min(loss_curve_values)}")




