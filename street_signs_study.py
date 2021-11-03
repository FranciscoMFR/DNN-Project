import os
import glob
from sklearn.model_selection import train_test_split
import shutil

from my_utils import create_generators, split_data, order_test_set

from deepLearningModels import street_signs_model



if __name__ == '__main__':

    path_to_train_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\training_data\\train"
    path_to_val_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\training_data\\val"
    path_to_test_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\Desktop\\UA\\Projeto\\DNN-Project\\archive\\Test"
    batch_size=64

    train_generator, val_generator, test_generator = create_generators(path_to_train_data, path_to_val_data, path_to_test_data)
    nbr_classes = train_generator.num_classes

    model = street_signs_model(nbr_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator,         
                epochs=3,
                batch_size=batch_size,
                validation_data=val_generator
            ) 