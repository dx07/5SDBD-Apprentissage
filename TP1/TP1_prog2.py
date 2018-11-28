from time import time

import numpy as np
from sklearn import neighbors

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, KFold

# ----------------------------------------------------

mnist = fetch_mldata('MNIST original')

indices = np.random.randint(70000, size=5000)
data = mnist.data[indices]
target = mnist.target[indices]

# ----------------- Test avec K = 10 -----------------

#
# xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8, test_size=0.2)

# clf = neighbors.KNeighborsClassifier(10)
#
# clf.fit(xtrain,ytrain)
# prediction = clf.predict(xtest)
# score = clf.score(xtest, ytest)
# print("Prédiction : {}, Valeur : {}, Efficacité : {}".format(prediction[4], ytest[4], score))

# ----------------- Boucle de variation de k -----------------

#
# xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8, test_size=0.2)

# for i in range (2,16):
#     clf = neighbors.KNeighborsClassifier(i)
#
#     clf.fit(xtrain, ytrain)
#     prediction = clf.predict(xtest)
#     score = clf.score(xtest, ytest)
#     print("K : {}, Efficacité : {}".format(i, score))

# ----------------- Boucle de variation de k avec KFold -----------------

# kf = KFold(14,shuffle=True)
# kf.get_n_splits(data)
#
# k = 2
#
# for train_index, test_index in kf.split(data):
#
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     xtrain, xtest = data[train_index], data[test_index]
#     ytrain, ytest = target[train_index], target[test_index]
#
#     clf = neighbors.KNeighborsClassifier(k)
#     clf.fit(xtrain,ytrain)
#     prediction = clf.predict(xtest)
#     score = clf.score(xtest, ytest)
#     print("K : {}, Efficacité : {}".format(k, score))
#
#     k = k + 1

# ----------------- Variation du pourcentage d'échantillon -----------------

# for i in range (2,10):
#
#     xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=(i/10))
#
#     clf = neighbors.KNeighborsClassifier(7)
#
#     clf.fit(xtrain, ytrain)
#     prediction = clf.predict(X = xtest)
#     score = clf.score(xtest, ytest)
#     print("Train size : {}%, Efficacité : {}".format(i/10, score))

# ----------------- Test avec n_jobs à 1 puis -1 -----------------

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

for i in [-1,1]:

    clf = neighbors.KNeighborsClassifier(7,n_jobs=i)

    clf.fit(xtrain, ytrain)
    time_start = time()
    prediction = clf.predict(xtest)
    time_stop = time()
    score = clf.score(xtest, ytest)
    print("n_jobs : {}, Temps total : {}".format(i,time_stop-time_start))
