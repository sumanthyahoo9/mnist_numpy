"""
Data preprocessing for the MNIST neural network
"""
import numpy as np
import matplotlib.pyplot as plt
from data_reader import (
    load_images, load_labels
)

# Load the images and visualize a few of them
def preprocess_images(filepath="data/raw/train-images-idx3-ubyte.gz"):
    """
    Load the images and visualize them
    """
    images = load_images(filepath=filepath)
    # Add a channel dimension for CNNs
    batches, height, width = images.shape
    images = images[:, None, :, :]
    assert images.shape == (batches, 1, height, width)
    images = images.astype(np.float32)
    # Normalize the images to the range [0, 1]
    images_normalized = images/255.0
    assert images_normalized.shape == (batches, 1, height, width)
    return images, images_normalized

# One-hot encode the labels
def one_hot_encode(filepath="data/raw/train-labels-idx1-ubyte.gz" , num_classes=10):
    """
    One-hot encode the labels
    """
    if filepath:
        labels_filepath = filepath
    else:
        labels_filepath = "data/raw/train-labels-idx1-ubyte.gz"
    labels = load_labels(filepath=labels_filepath)
    n_labels = labels.shape[0]
    one_hot_labels = np.zeros((n_labels, num_classes))
    one_hot_labels[np.arange(n_labels), labels] = 1
    print(f"The one hot encoded labels have the shape {one_hot_labels.shape}")
    print(f"\n The first few labels are \n {labels[:10]}")
    print(f"\n The first few rows are \n {one_hot_labels[:10]}")
    return labels, one_hot_labels

def visualize_images(images, labels, num_samples=10):
    """
    Load and visualize a few images
    images: [N, 1, 28, 28]
    labels: [N,] OR [N, 10]
    """
    n_images = images.shape[0]
    # Random indices
    indices = np.random.randint(0, n_images, (num_samples,))
    # Create a subplot grid
    _, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    for i, index in enumerate(indices):
        # Extract the image
        if len(images.shape) == 4:
            img = images[index, 0, :, :] # Remove the channel dim
        else:
            img = images[index]
        # Get the label
        if len(labels.shape) == 2:
            label = np.argmax(labels[index])
        else:
            label = labels[index]
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    images, _ = preprocess_images(filepath="data/raw/train-images-idx3-ubyte.gz")
    labels, _ = one_hot_encode(filepath="data/raw/train-labels-idx1-ubyte.gz")
    visualize_images(images=images, labels=labels)