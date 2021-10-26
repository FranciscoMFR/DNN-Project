import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D


# tensorflow.keras.Sequential
model = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(64, (2,2), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
    ]
)

# function approach : funtion that returns a model
# tensorflow.keras.Model : inherit from this class

def display_some_examples(examples, labels):

    plt.figure(figsize=(10, 10))

    for i in range(25):

        idx = np.random.randint(0, examples.shape[0]-1) 
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5, 5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()

if __name__=='__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    display_some_examples(x_train, y_train)