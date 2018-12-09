from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

mnist = fetch_mldata('MNIST original')
indices = np.random.randint(70000, size=5000)
data = mnist.data[indices]
target = mnist.target[indices]

# -------------------- Tests pour le TP (avec train à 80) ---------------------

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

# ----------------- Test avec K = 10 -----------------

'''

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8, test_size=0.2)

clf = neighbors.KNeighborsClassifier(10)

clf.fit(xtrain,ytrain)
prediction = clf.predict(xtest)
score = clf.score(xtest, ytest)
print("Prédiction : {}, Valeur : {}, Efficacité : {}".format(prediction[4], ytest[4], score))

'''

# ----------------- Boucle de variation de k -----------------

'''

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8, test_size=0.2)

for i in range (2,16):
    clf = neighbors.KNeighborsClassifier(i)

    clf.fit(xtrain, ytrain)
    prediction = clf.predict(xtest)
    score = clf.score(xtest, ytest)
    print("K : {}, Efficacité : {}".format(i, score))

'''

# ----------------- Test avec n_jobs à 1 puis -1 -----------------

'''
for i in [-1,1]:

    clf = neighbors.KNeighborsClassifier(7,n_jobs=i)

    clf.fit(xtrain, ytrain)
    time_start = time()
    prediction = clf.predict(xtest)
    time_stop = time()
    score = clf.score(xtest, ytest)
    print("n_jobs : {}, Temps total : {}".format(i,time_stop-time_start))
'''

# ----------------- Boucle de variation de k avec KFold -----------------

'''

kf = KFold(14,shuffle=True)
kf.get_n_splits(data)

k = 2

for train_index, test_index in kf.split(data):

    # print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = data[train_index], data[test_index]
    ytrain, ytest = target[train_index], target[test_index]

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(xtrain,ytrain)
    prediction = clf.predict(xtest)
    score = clf.score(xtest, ytest)
    print("K : {}, Efficacité : {}".format(k, score))

    k = k + 1

'''

# ----------------- Tests pour le rapport (avec train commun 70%) -----------------

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

# Variation de K

''' 
score=[]
execution_time=[]

values_of_k = range (2,16)

for k in values_of_k:

    clasifier = neighbors.KNeighborsClassifier(k)

    begin = time()
    clasifier.fit(xtrain,ytrain)
    predicted = clasifier.predict(X=xtest)
    end = time()
    total_time = end - begin

    score.append(clasifier.score(xtest, ytest))
    execution_time.append(total_time)
    print("score: ", clasifier.score(xtest, ytest))
    print("time: ", total_time)

plt.plot(values_of_k,execution_time)
plt.show()
plt.plot(values_of_k,score)
plt.show()
'''

# Variation n_jobs

'''
for i in [-1,1]:

    clf = neighbors.KNeighborsClassifier(3,n_jobs=i)

    clf.fit(xtrain, ytrain)
    time_start = time()
    prediction = clf.predict(xtest)
    time_stop = time()
    score = clf.score(xtest, ytest)
    print("n_jobs : {}, Temps total : {}".format(i,time_stop-time_start))
'''

# Matrice de confusion

''' 
clasifier = neighbors.KNeighborsClassifier(3)

begin = time()
clasifier.fit(xtrain,ytrain)
predicted = clasifier.predict(X=xtest)
end = time()

print("score: ", clasifier.score(xtest, ytest))

total_time = end - begin
print("time: ", total_time)

print(confusion_matrix(ytest,predicted))
'''

# Variation train / test

'''
score=[]
execution_time=[]

values_of_t = range (2,10)

for t in values_of_t:

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=(t/10))

    clasifier = neighbors.KNeighborsClassifier(3)

    begin = time()
    clasifier.fit(xtrain,ytrain)
    predicted = clasifier.predict(X=xtest)
    end = time()
    total_time = end - begin

    score.append(clasifier.score(xtest, ytest))
    execution_time.append(total_time)
    # print("score: ", clasifier.score(xtest, ytest))
    # print("time: ", total_time)

plt.plot(values_of_t,execution_time)
plt.show()
plt.plot(values_of_t,score)
plt.show()
'''
