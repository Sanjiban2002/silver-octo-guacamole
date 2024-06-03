#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

df = pd.concat([legitimate_df, phishing_df], axis=0)

df = df.sample(frac=1)

df = df.drop('URL', axis=1)
df = df.drop_duplicates()

X = df.drop('label', axis=1)
Y = df['label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Support Vector Machine
svm_model = svm.LinearSVC()

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# Gaussian Naïve Bayes
gnb_model = GaussianNB()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=60)

# k-Nearest Neighbors
knn_model = KNeighborsClassifier()

K = 5
total = X.shape[0]
index = int(total / K)

X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]

X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FP + FN)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall


svm_accuracy_list, svm_precision_list, svm_recall_list = [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
gnb_accuracy_list, gnb_precision_list, gnb_recall_list = [], [], []
rf_accuracy_list, rf_precision_list, rf_recall_list = [], [], []
knn_accuracy_list, knn_precision_list, knn_recall_list = [], [], []

for i in range(0, K):
    svm_model.fit(X_train_list[i], Y_train_list[i])
    svm_predictions = svm_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=svm_predictions).ravel()
    svm_accuracy, svm_precision, svm_recall = calculate_measures(tn, tp, fn, fp)
    svm_accuracy_list.append(svm_accuracy)
    svm_precision_list.append(svm_precision)
    svm_recall_list.append(svm_recall)

    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)

    gnb_model.fit(X_train_list[i], Y_train_list[i])
    gnb_predictions = gnb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=gnb_predictions).ravel()
    gnb_accuracy, gnb_precision, gnb_recall = calculate_measures(tn, tp, fn, fp)
    gnb_accuracy_list.append(gnb_accuracy)
    gnb_precision_list.append(gnb_precision)
    gnb_recall_list.append(gnb_recall)

    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall = calculate_measures(tn, tp, fn, fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)

    knn_model.fit(X_train_list[i], Y_train_list[i])
    knn_predictions = knn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=knn_predictions).ravel()
    knn_accuracy, knn_precision, knn_recall = calculate_measures(tn, tp, fn, fp)
    knn_accuracy_list.append(knn_accuracy)
    knn_precision_list.append(knn_precision)
    knn_recall_list.append(knn_recall)

SVM_accuracy = sum(svm_accuracy_list) / len(svm_accuracy_list)
SVM_precision = sum(svm_precision_list) / len(svm_precision_list)
SVM_recall = sum(svm_recall_list) / len(svm_recall_list)

print("Support Vector Machine accuracy ==> ", SVM_accuracy)
print("Support Vector Machine precision ==> ", SVM_precision)
print("Support Vector Machine recall ==> ", SVM_recall)

DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)

print("Decision Tree accuracy ==> ", DT_accuracy)
print("Decision Tree precision ==> ", DT_precision)
print("Decision Tree recall ==> ", DT_recall)

GNB_accuracy = sum(gnb_accuracy_list) / len(gnb_accuracy_list)
GNB_precision = sum(gnb_precision_list) / len(gnb_precision_list)
GNB_recall = sum(gnb_recall_list) / len(gnb_recall_list)

print("Gaussian Naïve Bayes accuracy ==> ", GNB_accuracy)
print("Gaussian Naïve Bayes precision ==> ", GNB_precision)
print("Gaussian Naïve Bayes recall ==> ", GNB_recall)

RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
RF_precision = sum(rf_precision_list) / len(rf_precision_list)
RF_recall = sum(rf_recall_list) / len(rf_recall_list)

print("Random Forest accuracy ==> ", RF_accuracy)
print("Random Forest precision ==> ", RF_precision)
print("Random Forest recall ==> ", RF_recall)

KNN_accuracy = sum(knn_accuracy_list) / len(knn_accuracy_list)
KNN_precision = sum(knn_precision_list) / len(knn_precision_list)
KNN_recall = sum(knn_recall_list) / len(knn_recall_list)

print("k-Nearest Neighbors accuracy ==> ", KNN_accuracy)
print("k-Nearest Neighbors precision ==> ", KNN_precision)
print("k-Nearest Neighbors recall ==> ", KNN_recall)

data = {'accuracy': [SVM_accuracy, DT_accuracy, GNB_accuracy, RF_accuracy, KNN_accuracy],
        'precision': [SVM_precision, DT_precision, GNB_precision, RF_precision, KNN_precision],
        'recall': [SVM_recall, DT_recall, GNB_recall, RF_recall, KNN_recall]
        }

index = ['SVM', 'DT', 'GNB', 'RF', 'KNN']

df_results = pd.DataFrame(data=data, index=index)

ax = df_results.plot.bar(rot=0)
plt.show()
