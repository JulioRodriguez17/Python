import numpy as np

#Funcion activacion sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivada de la funcion sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

#Datos de entrada y salida
X = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([[0],[1],[1],[0]])

#Inicializacion de pesos
np.random.seed(1)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

#Pesos de la capa de entrada
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

#Hiperparametros
epochs = 10000
learning_rate = 0.1

#Entrenamiento
for epoch in range(epochs):
    #FeedForward
    input_hidden = np.dot(X, weights_input_hidden) + bias_hidden
    output_hidden = sigmoid(input_hidden)

    input_output = np.dot(output_hidden, weights_hidden_output) + bias_output
    output = sigmoid(input_output)


#Retropopagacion
#Calculo de error
error = y - output

#Calculo de los deltas
delta_output = error * sigmoid_derivative(output)
error_hidden = delta_output.dot(weights_hidden_output.T)
delta_hidden = error_hidden * sigmoid_derivative(output_hidden)

#Actualizacion de pesos y bias
weights_hidden_output += output_hidden.T.dot(delta_output) * learning_rate
bias_output += np.sum(delta_output) * learning_rate
weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
bias_hidden += np.sum(delta_hidden) * learning_rate

#Imprimir resultados
print("Resultado despues del entrenamiento:")
print(output)