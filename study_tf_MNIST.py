import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.python.keras.backend import batch_dot


# tensorflow.keras.Sequential
seq_model = tf.keras.Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

# function approach : funtion that returns a model


def functional_model():

    my_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=my_input, outputs=x)

    return model


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


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = functional_model()

    # categorical_crossentropy -> use one hot coding
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics='accuracy')

    # model training
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

    # evaçuation on test set
    model.evaluate(x_test, y_test, batch_size=64)
