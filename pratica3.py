'''
SSC0270 - Redes Neurais
Prática 3 (aula 4)
Daniel Penna Chaves Bertazzo - 10349561

A biblioteca escolhida para o exercicio foi a sklearn
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


# Le o dataset
df = pd.read_csv("Dataset_3Cluster_4features.csv")

# Faz a divisao em conjunto de treino e conjunto de validacao
# 70% do dataset e' utilizado para treino
training_set, validation_set = train_test_split(df, test_size=0.3)

# Divide o conjunto de treinamento em entrada e saida
X_train = training_set.iloc[:, :-1].to_numpy()  # Dados de entrada
y_train = training_set.iloc[:, -1].to_numpy()   # Saida (label) esperada
# Divide o conjunto de validacao em entrada e saida
X_test = validation_set.iloc[:, :-1].to_numpy() # Dados de entrada
y_test = validation_set.iloc[:, -1].to_numpy()  # Saida (label) esperada

## Inicizalizando as MLPs 

# MLP 1: 2 camadas escondidas, uma com 10 neuronios e outra com 5. Ativacao sigmoide. Learning rate = 0.1
classifier1 = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, activation='logistic', learning_rate_init=0.1)

# MLP 2: uma camada escondida com 20 neuronios. Ativacao tangente hiperbolica. Learning rate = 0.001
classifier2 = MLPClassifier(hidden_layer_sizes=(20), max_iter=1000, activation='tanh', learning_rate_init=0.001)

# MLP 3: 3 camadas escondidas com 5 neuronios cada. Ativacao rectified linear unit. Learning rate = 0.1
classifier3 = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=1000, activation='relu', learning_rate_init=0.1)


# Treinando as MLPS
classifier1.fit(X_train, y_train)
classifier2.fit(X_train, y_train)
classifier3.fit(X_train, y_train)

# Testando as MLPS
y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)

def accuracy(C):
    '''
    A acuracia sera medida usando matriz de confusao. Dada a matriz de confusao C, temos:

    | C11  C12 . . . C1j |
    | C21  C22 . . . C2j |
    |  .    .  .      .  |
    |  .    .    .    .  |
    |  .    .      .  .  |
    | Ci1  Ci2 . . . Cij |

    onde Cij representa o numero de exepmlos do label i classificados como j. Com isso, temos (de forma simplificada):
            
          | casos onde a previsao foi feita corretamente, se i == j
    Cij = {
          | casos onde a previsao errou, se i != j
            
    Portanto, para calcular a acuracia, basta somarmos todos os elementos corretamente preditos (diagonal principal)
    e dividir pela soma de todos os elementos da matriz C.

    Input: C = matriz de confusao
    '''
    return C.trace() / C.sum()


# Calcula as matrizes de confusao
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)

# Calcula as acuracias
acc1 = accuracy(cm1)
acc2 = accuracy(cm2)
acc3 = accuracy(cm3)

# Imprime os resultados
print("Acuracia do modelo 1: ", acc1)
print("Acuracia do modelo 2: ", acc2)
print("Acuracia do modelo 3: ", acc3)