# import mathematical libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset from keras (mnist)
from keras.datasets import mnist

# load dataset (mnist)
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# number of unique train and test lables (0-9)
train_labels, train_counts = np.unique(y_train, return_counts=True)
train_dic = dict(zip(train_labels, train_counts))
test_labels, test_counts = np.unique(y_test, return_counts=True)
test_dic = dict(zip(test_labels, test_counts))

# check train and test dataset
print(f"Train dic: {train_dic}\nCount sum: {sum(train_dic.values())}\nTrain dataset size: {y_train.size}\n")
print(f"Test dic: {test_dic}\nCount sum: {sum(test_dic.values())}\nTest dataset size: {y_test.size}\n")

# train and test dataset histograms
train_lst_labels = np.array(list(test_dic.values()))/sum(test_dic.values())*100
train_lst_labels = np.round(train_lst_labels, 1)
train_perc_labels=[]
for idx,val in enumerate(train_lst_labels):
    train_perc_labels.append(f"{str(val)}%")
plt.figure(num=1)
train = plt.bar(x=train_dic.keys(), height=train_dic.values(), edgecolor="black")
plt.bar_label(train, labels=train_perc_labels, label_type="edge")
plt.title("Training Dataset Histogram")
plt.xlabel("Training Labels")
plt.ylabel("Number of training examples")
test_lst_labels = np.array(list(test_dic.values()))/sum(test_dic.values())*100
test_lst_labels = np.round(test_lst_labels, 1)
test_perc_labels=[]
for idx,val in enumerate(test_lst_labels):
    test_perc_labels.append(f"{str(val)}%")
#print(test_perc_labels)
plt.figure(num=2)
test = plt.bar(x=test_dic.keys(), height=test_dic.values(), edgecolor="black")#, width=0.7)
plt.bar_label(test, labels=test_perc_labels, label_type="edge")
plt.title("Test Dataset Histogram")
plt.xlabel("Test Labels")
plt.ylabel("Number of test examples")
plt.show()