import shutil

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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# image dimensions (assumed square) 
image_size = x_train.shape[1]
input_size = image_size**2

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype("float32") / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype("float32") / 255

# data shape
print(f"x_train shape:{x_train.shape}")

# network parameters
filename = "shallow_mnist"
batch_size = 32
lr = 0.1
epochs = 50
momento = 0.1
hidden_units = 800
dropout = 0.45            #não utilizado

# model is a shallow network with softmax and dropout
model = Sequential()
model.add(Flatten(input_dim=input_size))
model.add(Dense(hidden_units, activation="relu"))
model.add(Dense(num_labels, activation="softmax"))

model.summary()

# save the model summary to file
with open(filename + "_report.txt", "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n")) # pass the file handle in as a lambda function to make it callable

# create the model image
plot_model(model, to_file="shallow-mnist.png", show_shapes=True)

# delete the training end validation folders
shutil.rmtree("./logs/shallow_mnist/train", ignore_errors=True)
shutil.rmtree("./logs/shallow_mnist/validation", ignore_errors=True)

# edit the tensorboard callback
tensorboard_callback = TensorBoard(log_dir="./logs/shallow_mnist",
                                   histogram_freq=0,                 #frequency (in epochs) at which to compute weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
                                   write_graph=True,                 #whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
                                   write_images=False,               #whether to write model weights to visualize as image in TensorBoard.
                                   update_freq="batch",              #'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000, the callback will write the metrics and losses to TensorBoard every 1000 batches. Note that writing too frequently to TensorBoard can slow down your training.
                                   profile_batch=0,                  #Profile the batch(es) to sample compute characteristics. profile_batch must be a non-negative integer or a tuple of integers. A pair of positive integers signify a range of batches to profile. By default, profiling is disabled.
                                   embeddings_freq=0,                #frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings won't be visualized.
                                   embeddings_metadata=None)         #Dictionary which maps embedding layer names to the filename of a file in which to save metadata for the embedding layer. In case the same metadata file is to be used for all embedding layers, a single filename can be passed.

# creating the Stochastic Gradient Descent optimizer
sgd=SGD(lr=lr,
        decay=1e-6,
        momentum=momento,
        nesterov=True)

# Compiling the model with compile()
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])

# Training the model with fit()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[tensorboard_callback])

#Evaluating model performance with evaluate()
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"\nTest accuracy({batch_size}): {(100.0 * acc):.2f}%")
