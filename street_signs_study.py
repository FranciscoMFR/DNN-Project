import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from my_utils import create_generators, split_data, order_test_set

from deepLearningModels import street_signs_model
import tensorflow as tf


if __name__ == '__main__':

    path_to_train_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\training_data\\train"
    path_to_val_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\training_data\\val"
    path_to_test_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\Desktop\\UA\\Projeto\\DNN-Project\\archive\\Test"
    batch_size=32
    epochs = 1000
    lr = 0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train_data, path_to_val_data, path_to_test_data)
    nbr_classes = train_generator.num_classes


    TRAIN = True
    TEST = False

    if TRAIN:
        path_to_save_model = "./Models"
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

        model = street_signs_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,         
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=val_generator,
                    callbacks=[ckpt_saver, early_stop]
                ) 

    if TEST:

        model = tf.keras.models.load_model('./Models')
        model.summary()

        print('Evaluating validation set:')
        model.evaluate(val_generator)

        print('Evaluating test set:')
        model.evaluate(test_generator)

        #0.99923 asdasdasdasdasdasd