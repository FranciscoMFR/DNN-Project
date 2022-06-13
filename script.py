# import mathematical libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset from keras (mnist)
from keras.datasets import mnist

# import model things
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.regularizers import l2

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# load dataset (mnist)
(x_train, y_train),(x_test,y_test) = mnist.load_data()

# number of unique train lables (0-9)  TO DO construct an histogram
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

# number of unique test lables (0-9)  TO DO construct an histogram
unique, counts = np.unique(y_test, return_counts=True)
print("Test labels: ", dict(zip(unique, counts)))

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

# plot the mnist sample
plt.figure(figsize=(5, 5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i+1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.show()
plt.savefig("mnist-samples.png")
plt.close("all")

# compute the number of labels
num_labels = len(np.unique(y_train))

# convert to one-hot vector to use categorical entropy
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# image dimensions (assumed square)

image_size = x_train.shape[1]
input_size = image_size**2
print(input_size)

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype("float32") / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype("float32") / 255

# network parameters
batch_size = 32
hidden_units = 800
dropout = 0.45        #dropout is the dropout rate (sections 7 - Overfitting and Regularization)

# model is a 3-layer MLP with ReLu and dropout 
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
#model.add(Dense(hidden_units, kernel_regularizer=l2(0.001), input_dim=input_size))  #l2 weight regularizer with fraction=0.001
model.add(Activation("relu"))
#model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation("relu"))
#model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation("softmax"))

model.summary()

plot_model(model, to_file="mlp-mnist.png", show_shapes=True)

# Copiling the model with compile()
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Trainning the model with fit()
model.fit(x_train, y_train, epochs=320, batch_size=batch_size)

#Evaluating model performance with evaluate()
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


