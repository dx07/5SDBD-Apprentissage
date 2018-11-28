import random
from time import time

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')

mnist.data, mnist.target = shuffle(mnist.data, mnist.target)

xtrain, xtest, ytrain, ytest = train_test_split(mnist.data, mnist.target, train_size=0.7)

# -------------- Test simple avec 50 neurones en 1 couche ----------------

# mlpc = MLPClassifier(hidden_layer_sizes=50)
#
# print("Step 1 : Fit")
# mlpc.fit(xtrain,ytrain)
# print("Step 2 : Prédiction")
# prediction = mlpc.predict(xtest)
# print("Step 3 : Score")
# score = mlpc.score(xtest, ytest)
# print("Step 4 : Précision")
# precision = metrics.precision_score(ytest, prediction,average='micro')
# print("Prédiction : {}, Valeur : {}, Efficacité : {}, Précision : {}".format(prediction[4], ytest[4], score, precision))

# ------------------ De 2 à 20 couches de 50 neurones ---------------------

# list = [50]
#
# for i in range (2,20):
#
#     list.append(50)
#
#     mlpc = MLPClassifier(hidden_layer_sizes=tuple(list))
#
#     print("Step 1 : Fit")
#     mlpc.fit(xtrain,ytrain)
#     print("Step 2 : Prédiction")
#     prediction = mlpc.predict(xtest)
#     print("Step 3 : Score")
#     score = mlpc.score(xtest, ytest)
#     print("Step 4 : Précision")
#     precision = metrics.precision_score(ytest, prediction,average='micro')
#     print("Prédiction : {}, Valeur : {}, Efficacité : {}, Précision : {}".format(prediction[4], ytest[4], score, precision))

# -------------------- Couches aléatoires -------------------

# for modeles in range (1,6):
#
#     couches = random.randint(1,11)
#
#     list = []
#
#     for couche in range (1,couches+1):
#
#         list.append(random.randint(10,301))
#
#     mlpc = MLPClassifier(hidden_layer_sizes=tuple(list))
#
#     time_start = time()
#     print("Step 1 : Fit")
#     mlpc.fit(xtrain, ytrain)
#     print("Step 2 : Prédiction")
#     prediction = mlpc.predict(xtest)
#     print("Step 3 : Score")
#     score = mlpc.score(xtest, ytest)
#     print("Step 4 : Précision")
#     precision = metrics.precision_score(ytest, prediction, average='micro')
#     time_stop = time()
#
#     print("Réseau : {}\nEfficacité : {}, Précision : {}, Temps : {}".format(list,score,precision,time_stop-time_start))

# ------------------------------- Convergence ----------------------------

for solver in ['lbfgs','sgd','adam']:

    mlpc = MLPClassifier(hidden_layer_sizes=50, solver=solver)

    time_start = time()
    print("Step 1 : Fit")
    mlpc.fit(xtrain,ytrain)
    print("Step 2 : Prédiction")
    prediction = mlpc.predict(xtest)
    print("Step 3 : Score")
    score = mlpc.score(xtest, ytest)
    print("Step 4 : Précision")
    precision = metrics.precision_score(ytest, prediction,average='micro')
    time_stop = time()

    print("Solver : {}, Efficacité : {}, Précision : {}, Temps : {}".format(solver,score,precision,time_stop-time_start))


