import pandas as pd 
import numpy as np

# Le o arquivo
df = pd.read_csv('Aula2-exec1.csv')

# Saidas esperadas
y = df.iloc[:,-1].to_numpy()

# Input
x = df.iloc[:,0:-1].to_numpy()

# Gera os pesos aleatoriamente
w = np.random.uniform(low=-1.0, high=1.0, size=(2,))
# Gera os bias aleatoriamente
bias = np.random.uniform(low=-1.0, high=1.0, size=(200,))

# Faz a soma das multiplicacoes
weighted_sum = x.dot(w)
# Soma com os bias
weighted_sum = np.add(weighted_sum, bias) # weighted_sum armazena a soma antes da ativacao

# Funcao limiar
def limiar_activation(weighted_sum):
  return [1.0 if result >= 0.0 else 0.0 for result in np.nditer(weighted_sum)]

# Funcao linear
def linear_activation(weighted_sum):
  linear = []
  for result in np.nditer(weighted_sum):
    if result <= -0.5:
      linear.append(0.0)
    elif result > -0.5 and result < 0.5:
      linear.append(result + 0.5)
    else:
      linear.append(1.0)

  return linear

# Funcao sigmoide
def sigmoid_activation(weighted_sum):
  return (1 / (1 + np.exp(-weighted_sum))).tolist()

# Realiza a classificacao
def classify(y_hat):
  return [1 if x <= 0.5 else 2 for x in y_hat]

# Calcula acuracia
def accuracy(y, y_hat):
  correct = 0
  for i, result in enumerate(y_hat):
    if result == y[i]:
      correct += 1
  
  # Acuracia = numero de acertos / numero total de instancias
  return correct / len(y)

# Realiza a ativacao com as 3 funcoes
limiar  = limiar_activation(weighted_sum)
linear  = linear_activation(weighted_sum)
sigmoid = sigmoid_activation(weighted_sum)

# Classifica os resultados
limiar_result  = classify(limiar)
linear_result  = classify(linear)
sigmoid_result = classify(sigmoid)

# Calcula a acuracia para cada funcao de ativacao
limiar_acc  = accuracy(y, limiar_result)
linear_acc  = accuracy(y, linear_result)
sigmoid_acc = accuracy(y, sigmoid_result)

# Imprime os resultados
print("Limiar accuracy:  ", limiar_acc)
print("linear accuracy:  ", linear_acc)
print("sigmoid accuracy: ", sigmoid_acc)