import os
import glob
from sklearn.model_selection import train_test_split
import shutil

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):

    folders = os.listdir(path_to_data)

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))
        print(folder)

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)



if __name__ == '__main__':

    path_to_data = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\Train"
    path_to_save_train = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\training_data\\train"
    path_to_save_val = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\training_data\\val"
split_data(path_to_data, path_to_save_train, path_to_save_val)