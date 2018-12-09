import numpy
from time import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

mnist = fetch_mldata('MNIST original')
indices = numpy.random.randint(70000, size=5000)
data = mnist.data[indices]
target = mnist.target[indices]

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size= 0.7)

''' temps d'execution

score=[]
execution_time=[]

values_of_c = ['sigmoid','rbf','linear','poly']

for c in values_of_c:

    clasifier = svm.SVC(kernel=c)

    begin = time()
    clasifier.fit(xtrain,ytrain)
    predicted = clasifier.predict(X=xtest)
    end = time()

    total_time = end - begin
    execution_time.append(total_time)
    print("temps d'execution: ", total_time)

    # print("time: ", total_time)

plt.plot(values_of_c,execution_time)
plt.show()
'''

''' score

score=[]
execution_time=[]

values_of_c = ['sigmoid','rbf','linear','poly']

for c in values_of_c:

    clasifier = svm.SVC(kernel=c)

    begin = time()
    clasifier.fit(xtrain,ytrain)
    predicted = clasifier.predict(X=xtest)
    end = time()

    score.append(clasifier.score(xtest, ytest))
    print("score: ", clasifier.score(xtest, ytest))

    #total_time = end - begin
    #print("time: ", total_time)

plt.plot(values_of_c,score)
plt.show()
'''

'''
#variation C

score=[]
execution_time=[]

values_of_c = numpy.linspace(0.1,1, 5, endpoint=True)

for c in values_of_c:

    clasifier = svm.SVC(kernel='rbf', C=c)

    begin = time()
    clasifier.fit(xtrain,ytrain)
    predicted = clasifier.predict(X=xtest)
    end = time()
    total_time = end - begin

    score.append(clasifier.score(xtest, ytest))
    execution_time.append(total_time)
    # print("score: ", clasifier.score(xtest, ytest))
    # print("time: ", total_time)

plt.plot(values_of_c,execution_time)
plt.show()
plt.plot(values_of_c,score)
plt.show()
'''

''''# variation gamma

score=[]
execution_time=[]

values_of_c = numpy.linspace(0.1,1, 5, endpoint=True)

for c in values_of_c:

    clasifier = svm.SVC(kernel='rbf', gamma=c)

    begin = time()
    clasifier.fit(xtrain,ytrain)
    predicted = clasifier.predict(X=xtest)
    end = time()
    total_time = end - begin

    score.append(clasifier.score(xtest, ytest))
    execution_time.append(total_time)
    # print("score: ", clasifier.score(xtest, ytest))
    # print("time: ", total_time)

plt.plot(values_of_c,score)
plt.show()
plt.plot(values_of_c,execution_time)
plt.show()
'''

'''avec les paramètres générés tout seul    
parameters = {'kernel':('linear', 'rbf', 'sigmoid', 'poly'), 'C':[1, 10]}

svc = svm.SVC(kernel='sigmoid')

clasifier = GridSearchCV(svc,parameters)
begin = time()
clasifier.fit(xtrain, ytrain)
predicted = clasifier.predict(X=xtest)
end = time()

score.append(clasifier.score(xtest, ytest))
print("score: ", clasifier.score(xtest, ytest))

total_time = end - begin
print("time: ", total_time)
'''

''' matrice de confusion
clasifier = svm.SVC(kernel='linear')

begin = time()
clasifier.fit(xtrain,ytrain)
predicted = clasifier.predict(X=xtest)
end = time()

score.append(clasifier.score(xtest, ytest))
print("score: ", clasifier.score(xtest, ytest))

total_time = end - begin
print("time: ", total_time)

print(confusion_matrix(ytest,predicted))
'''