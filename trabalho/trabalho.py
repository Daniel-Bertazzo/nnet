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
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

# # PARTE 1 - IRIS

# # MLP - 2 camadas: uma com 5 neuronios e outra com 3
# mlp_iris = MLPClassifier(hidden_layer_sizes=(5, 3), activation='logistic',
#                          learning_rate='constant', learning_rate_init=0.01, max_iter=1000)

# # SVM com kernel radial
# svm_iris = svm.SVC(kernel='rbf', tol=1e-4, max_iter=1000)

# # Faz a cross-validation, treino e teste
# results_mlp_iris = cross_validate(mlp_iris, X, y, cv=10, return_train_score=True)
# results_svm_iris = cross_validate(svm_iris, X, y, cv=10, return_train_score=True)

# print(np.mean(results_mlp_iris['train_score']), np.mean(results_mlp_iris['test_score']))
# print(np.mean(results_svm_iris['train_score']), np.mean(results_svm_iris['test_score']))

# PARTE 2 - MNIST
mnist = load_digits()
X = mnist.data
y = mnist.target

# mlp_mnist = MLPClassifier(hidden_layer_sizes=(400, 400), activation='logistic',
#                           learning_rate='constant', learning_rate_init=0.01, max_iter=1000)

svm_mnist = svm.SVC(kernel='rbf', tol=1e-5, max_iter=-1)

# Faz a cross-validation, treino e teste
# results_mlp_mnist = cross_validate(mlp_mnist, X, y, cv=10, return_train_score=True)
results_svm_mnist = cross_validate(svm_mnist, X, y, cv=10, return_train_score=True)

# print(np.mean(results_mlp_mnist['train_score']), np.mean(results_mlp_mnist['test_score']))
print(np.mean(results_svm_mnist['train_score']), np.mean(results_svm_mnist['test_score']))