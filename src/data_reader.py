"""
Download and save the data
"""
import os
import gzip
import urllib.request
import numpy as np

def download_mnist(data_dir='data/raw'):
    """
    Download the MNIST datasets
    """
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    os.makedirs(data_dir, exist_ok=True)
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}")
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)
        else:
            continue

def load_images(filepath):
    """
    Load the images from the MNIST binary format
    """
    with gzip.open(filepath, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        # Read the pixel data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = np.reshape(data, (num_images, rows, cols))
        return images

def load_labels(filepath):
    """
    Load the labels
    """
    with gzip.open(filepath, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")
        # Read the pixel data
        data = np.frombuffer(f.read(num_labels), dtype=np.uint8)
        return data

if __name__ == "__main__":
    download_mnist()
    train_images = load_images("data/raw/train-images-idx3-ubyte.gz")
    print(f"The training set has the shape {train_images.shape}")
    train_labels = load_labels("data/raw/train-labels-idx1-ubyte.gz")
    print(f"There are {train_labels.shape} labels")