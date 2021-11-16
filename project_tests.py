from os import name
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import learning_phase
from deepLearningModels import mnist_model


if __name__=='__main__':

   batch_size = 32
   epochs = 1000
   lr = 1.0

   nbr_classes = 10

   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

   x_train = x_train.astype('float32') / 255
   x_test = x_test.astype('float32') / 255

   #print("x_train.shape = ", x_train.shape) (60000, 28, 28)  

   x_train = np.expand_dims(x_train, axis=-1)
   x_test = np.expand_dims(x_test, axis=-1)

   #print("x_train.shape = ", x_train.shape)  (60000, 28, 28, 1)

   TRAIN = True
   TEST = False

   if TRAIN:
       path_to_save_model = "./Models/projectModel"
       ckpt_saver = ModelCheckpoint(
           path_to_save_model,
           monitor="val_accuracy",
           mode="max",
           save_best_only=True,
           save_freq="epoch",
           verbose=1
       )

       #early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

       model = mnist_model(nbr_classes)

       optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, name="SGD")

       model.compile(
           optimizer=optimizer,
           loss="sparse_categorical_crossentropy",
           metrics=['accuracy']
       )

       model.fit(x_train, 
                y_train, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_split=0.2,
                callbacks=[ckpt_saver]
       )
