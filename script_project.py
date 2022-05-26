# import mathematical libs
from telnetlib import SE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset from keras (mnist)
from keras.datasets import mnist

# import model things
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import TensorBoard
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.regularizers import l2
import tensorboard

#load dataset (mnist)
(x_train, y_train),(x_test,y_test) =mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))

# convert to one-hot vector to use categorical entropy
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

# image dimensions (assumed square) 
image_size = x_train.shape[1]
input_size = image_size**2

# resize and normalize

x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype("float32") / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype("float32") / 255

# network parameters
filename = "shallow_mnist"
batch_size = 32
lr = 1
epochs = 50
momento = 0.9
hidden_units = 800
dropout = 0.45

# model is a shallow network with softmax and dropout
model = Sequential()
model.add(Flatten(input_dim=input_size))
#model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation("relu"))
#model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation("relu"))
#model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation("softmax"))

model.summary()

# save the model summary to file
with open(filename + "_report.txt", "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n")) # pass the file handle in as a lambda function to make it callable

# create the model image
plot_model(model, to_file="shallow-mnist.png", show_shapes=True)

# edit the tensorboard callback
tensorboard_callback = TensorBoard(log_dir="./logs/shallow_mnist")

# creating the Stochastic Gradient Descent optimizer
#sgd=SGD(learning_rate=lr,
#        momentum=momento,
#        name="SGD")

# Compiling the model with compile()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Training the model with fit()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[tensorboard_callback])