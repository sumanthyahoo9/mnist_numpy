"""
Load the training and test data
"""
from preprocess_data import preprocess_images, one_hot_encode

def get_data():
    """
    Get the training and test data, along with the labels
    """
    _, x_train = preprocess_images()
    _, x_test = preprocess_images(filepath="data/raw/t10k-images-idx3-ubyte.gz")
    _, y_train = one_hot_encode()
    _, y_test = one_hot_encode(filepath="data/raw/t10k-labels-idx1-ubyte.gz")
    assert x_train.shape == (60000, 1, 28, 28)
    assert x_test.shape == (10000, 1, 28, 28)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    return x_train, x_train, x_test, y_test

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_data()
    print("\n")
    print(f"There are {train_images.shape[0]} images for training and {train_labels.shape[0]} as the labels")
    print(f"Each training data image is of shape {train_images[1].shape}")
    print(f"There are {test_images.shape[0]} images for training and {test_labels.shape[0]} as the labels")
    print(f"Each testing data image is of shape {test_images[1].shape}")
