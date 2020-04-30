''' 
SCC0270 - Redes Neurais e Aprendizado Profundo
Trabalho Avaliativo - 1

Grupo:
    Vinicius Torres Dutra Maia da Costa - nUSP: 10262781
    Daniel Penna Chaves Bertazzo        - nUSP: 10349561
    Alexandre Norcia Medeiros           - nUSP: 10295583

Requisitos para rodar o codigo:
    python - versao 3 
    bibliotecas: numpy, pandas, matplotlib, sklearn 
'''
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def custom_confusion_matrix(X, Y, model):
    y_pred = model.predict(X)
    return confusion_matrix(Y, y_pred)

def custom_cross_validation(X, Y, model, kfolds=10):
    cv_results = cross_validate(model, X, Y, cv=kfolds, return_train_score=True, return_estimator=True)
    n_classes = np.unique(Y).size   
    confMats = np.zeros((n_classes, n_classes))
    for estimator in cv_results["estimator"]:
        confMats += custom_confusion_matrix(X, Y, estimator)

    cv_results["conf_matrix"] = confMats/kfolds
    return cv_results

# ******************************************************* PARTE 1 - IRIS *******************************************************

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

# MLP - 2 camadas: uma com 5 neuronios e outra com 3
mlp_iris = MLPClassifier(hidden_layer_sizes=(5, 3), activation='logistic',
                         learning_rate='constant', learning_rate_init=0.01, max_iter=2000)
# SVM com kernel radial
svm_iris = svm.SVC(kernel='rbf', tol=1e-4, max_iter=-1)

# Faz a cross-validation, treino e teste

results_mlp_iris = custom_cross_validation(X, y, mlp_iris)
results_svm_iris = custom_cross_validation(X, y, svm_iris)

conf_interval = 1.96/np.sqrt(10)
with open('results_iris.txt', 'a') as f:

    f.write("---- MLP - 2 hidden layer of size = (5, 3), activation function = logistic, learning rate = 0.01 ----\n")
    f.write("Train accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_mlp_iris['train_score']), np.std(results_mlp_iris['train_score']) * conf_interval))
    f.write("Validation accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_mlp_iris['test_score']), np.std(results_mlp_iris['test_score']) * conf_interval))
    f.write("Confusion matrix:\n")
    np.savetxt(f, results_mlp_iris["conf_matrix"], fmt='%.3f')

    f.write("\n---- SVM non-linear(soft-margin) - kernel = radial basis function ----\n")
    f.write("Train accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_svm_iris['train_score']), np.std(results_svm_iris['train_score']) * conf_interval))
    f.write("Validation accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_svm_iris['test_score']), np.std(results_svm_iris['test_score']) * conf_interval))
    f.write("Confusion matrix:\n")
    np.savetxt(f, results_svm_iris["conf_matrix"], fmt='%.3f')


# ******************************************************* PARTE 2 - MNIST *******************************************************

mnist = load_digits()
X = mnist.data
y = mnist.target

mlp_mnist = MLPClassifier(hidden_layer_sizes=(400, 400), activation='logistic',
                          learning_rate='constant', learning_rate_init=0.01, max_iter=1000)

svm_mnist = svm.SVC(kernel='poly', degree=3, tol=1e-5, max_iter=-1)

# Faz a cross-validation, treino e teste
results_mlp_mnist = custom_cross_validation(X, y, mlp_mnist)
results_svm_mnist = custom_cross_validation(X, y, svm_mnist)

conf_interval = 1.96/np.sqrt(10)
with open('results_mnist.txt', 'a') as f:

    f.write("---- MLP - 2 hidden layer of size = (400, 400), activation function = logistic, learning rate = 0.01 ----\n")
    f.write("Train accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_mlp_mnist['train_score']), np.std(results_mlp_mnist['train_score']) * conf_interval))
    f.write("Validation accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_mlp_mnist['test_score']), np.std(results_mlp_mnist['test_score']) * conf_interval))
    f.write("Confusion matrix:\n")
    np.savetxt(f, results_mlp_mnist["conf_matrix"], fmt='%.3f')

    f.write("\n---- SVM non-linear(soft-margin) - kernel = polynomial with degree = 3 ----\n")
    f.write("Train accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_svm_mnist['train_score']), np.std(results_svm_mnist['train_score']) * conf_interval))
    f.write("Validation accuracy: %0.3f (+/- %0.3f)\n" % (np.mean(results_svm_mnist['test_score']), np.std(results_svm_mnist['test_score']) * conf_interval))
    f.write("Confusion matrix:\n")
    np.savetxt(f, results_svm_mnist["conf_matrix"], fmt='%.3f')
