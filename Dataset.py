import os
import numpy as np

from keras.src import backend
from keras.src.datasets.cifar import load_batch
from keras.src.utils.file_utils import get_file

def load_data():
    """The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | airplane    |
    |   1   | automobile  |
    |   2   | bird        |
    |   3   | cat         |
    |   4   | deer        |
    |   5   | dog         |
    |   6   | frog        |
    |   7   | horse       |
    |   8   | ship        |
    |   9   | truck       |
    """
    dirname = "cifar-10-batches-py-target"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(
        fname=dirname,
        origin=origin,
        extract=True,
        file_hash=(  # noqa: E501
            "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
        ),
    )

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    # batches are within an inner folder
    path = os.path.join(path, "cifar-10-batches-py")
    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)