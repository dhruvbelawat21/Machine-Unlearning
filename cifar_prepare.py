import sys
sys.path.append('../')
import numpy as np
import tensorflow_datasets as tfds
data_dir = '../.data'
cifar = tfds.load('cifar10', data_dir=data_dir)
x_train, y_train = list(zip(*((sample['image'], sample['label']) for sample in cifar['train'])))
x_train = np.stack(x_train)
y_train = np.stack(y_train)
x_test, y_test = list(zip(*((sample['image'], sample['label']) for sample in cifar['test'])))
x_test = np.stack(x_test)
y_test = np.stack(y_test)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


n_classes = 10
y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)
y_val = to_categorical(y_val, num_classes=n_classes)
from conf import BASE_DIR

data_dir = BASE_DIR/'train_test_data'/'Cifar-test'
data_dir.mkdir(parents=True, exist_ok=True)

for arr, filename in [
        (x_train, 'x_train.npy'),
        (y_train, 'y_train.npy'),
        (x_test, 'x_test.npy'),
        (y_test, 'y_test.npy'),
        (x_val, 'x_valid.npy'),
        (y_val, 'y_valid.npy')]:
    np.save(data_dir/filename, arr)