import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ********************************* ..:: Classe Neuron ::.. *********************************
class Neuron:
    def __init__(self, dataset, neuron_type):
        # Gera os pesos iniciais aleatoriamente (vetor Nx1, onde N = numero de features)
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(df.shape[1] - 1,))
        # Gera os bias iniciais aleatoriamente (unico valor)
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=(1,))
        self.type = neuron_type

    def train(self, x, y):
        # Inicializa o erro acumulado com um valor maior que 0.5 (simplesmente para entrar no loop)
        accumulated_error = 1.0
        
        # Variaveis utilizadas no plot dos erros acumulados
        errors = []
        j = 0

        while accumulated_error > 0.5:
            accumulated_error = 0.0
            # percorre o dataset inteiro (linhas)
            for i in range(x.shape[0]):
                # Calcula a soma ponderada dos pesos com as entradas
                weighted_sum = feed_forward(x[i,:], self.weights, self.bias)
                
                # Realiza a ativacao do resultado
                if self.type == 'perceptron':
                    output = limiar_activation(weighted_sum) + 1
                elif self.type == 'adaline':
                    output = linear_activation(weighted_sum) + 1
                else:
                    print("Error: invalid neuron type")

                if output != y[i]:
                    self.weights -= eta * x[i,:] * (output - y[i])
                    self.bias -= eta * (output - y[i])
                
                accumulated_error += abs(output - y[i])
            
            # Imprime o erro acumulado dessa iteracao
            print(accumulated_error)

            # Variaveis utilizadas no plot dos erros acumulados
            errors.append(accumulated_error)
            j += 1

        # Plotando a evolucao do erro
        iterations = np.arange(0, j)
        plt.plot(iterations, np.asarray(errors), 'o-')
        plt.xlabel('Número de iterações')
        plt.ylabel('Erro acumulado')
        plt.show()


        
# ********************************* ..:: Funcoes ::.. *********************************
def feed_forward(x, weights, bias):
    return x.dot(weights) + bias

def linear_activation(weighted_sum):
        if weighted_sum <= -0.5:
            return 0
        elif weighted_sum > -0.5 and weighted_sum < 0.5:
            return weighted_sum + 0.5
        else:
            return 1

def limiar_activation(weighted_sum):
    return 0 if weighted_sum < 0.5 else 1

# ********************************* ..:: Lendo o arquivo ::.. *********************************

# Le o arquivo
df = pd.read_csv("Aula3-dataset_2.csv")

 # Saidas esperadas (ultima coluna do dataset)
y = df.iloc[:,-1].to_numpy()
# Input
x = df.iloc[:, 0:-1].to_numpy()

# ********************************* ..:: Main ::.. *********************************

# Define o learning rate
eta = 0.1

# Inicializa os neuronios
perceptron = Neuron(df, 'perceptron')
adaline    = Neuron(df, 'adaline')

'''
Treina os neuronios e plota a evolucao dos erros
OBS.: devem ser rodados separadamente
 '''
# perceptron.train(x, y)
adaline.train(x, y)