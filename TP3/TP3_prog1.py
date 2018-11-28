from time import time

import numpy as np
from sklearn import svm
from sklearn.utils import shuffle

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')

# indices = np.random.randint(70000, size=10000)
# mnist.data = mnist.data[indices]
# mnist.target = mnist.target[indices]

mnist.data = mnist.data[0:10000]
mnist.target = mnist.target[0:10000]

mnist = fetch_mldata('MNIST original')
mnist.data, mnist.target = shuffle(mnist.data, mnist.target)

xtrain, xtest, ytrain, ytest = train_test_split(mnist.data, mnist.target, train_size=0.7)

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    clsvm = svm.SVC(kernel=kernel)
    time_start = time()

    print("Step 1 : Fit")
    clsvm.fit(xtrain,ytrain)
    print("Step 2 : Prédiction")
    prediction = clsvm.predict(xtest)
    print("Step 3 : Score")
    score = clsvm.score(xtest, ytest)

    print("Kernel : {}, Temps : {}, Prédiction : {}, Valeur : {}, Efficacité : {}".format(kernel, time()-time_start, prediction[4], ytest[4], score))

