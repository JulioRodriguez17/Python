#Se importan las bibliotecas a utilizar
from sklearn import datasets 
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
escalar = StandardScaler()


#Importamos los datos de la misma biblioteca de scikit-learn
dataset = datasets.load_breast_cancer()
print('Dataset:')
print(dataset) 

#Para entender mejor los datos, añadimos la instruccion keys
print('Información en el dataset:')
print(dataset.keys()) 

# Imprimir los datos contenidos en la llave 'data'
print('Datos:')
print(dataset['data'])

#Imprimir los datos contenidos en la llave 'target'
print('Datos de target:')
print(dataset['target'])

#Imprimir los datos contenidos en la llave 'target_names'
print('Nombres de las etiquetas:')
print(dataset['target_names'])

#Imprimir los datos contenidos en la llave 'DESCR'
print('Descripción del conjunto de datos:')
print(dataset['DESCR'])

#Imprimir los datos contenidos en la llave 'feature_names'
print('Nombres de las características:')
print(dataset['feature_names'])

# Verificar la cantidad de datos
n_datos = dataset.data.shape[0]
print("Cantidad de datos en el conjunto de datos:", n_datos)

# Verificar el número de atributos/características
n_atributos = dataset.data.shape[1]
print("Número de atributos/características:", n_atributos)

# Verificar que todos los atributos/características son numéricos
# Para esto, podemos verificar el tipo de dato de las características
if np.issubdtype(dataset.data.dtype, np.float64):
    print("Todos los datos en dataset.data son numéricos.")
else:
    print("No todos los datos en dataset.data son numéricos.")

#Seleccionamos todas las columnas
X = dataset.data 

#Defino los datos correspondientes a las etiquetas
y = dataset.target 

#Separo los datos en entrenamiento y prueba para probar los algoritmos 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print("X_train: \n", X_train)
#print("X_test: \n", X_test)
#print("y_train: \n", y_train)
#print("y_test: \n", y_test)

X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test) 

#print("X_train: \n", X_train)
#print("X_test: \n", X_test)

#Naive Bayes 
from sklearn.naive_bayes import GaussianNB 
algoritmo = GaussianNB() 

#Entrenamiento del modelo 
algoritmo.fit(X_train, y_train) 

#Realizo una predicción 
y_pred = algoritmo.predict(X_test)

#Verifico la matriz de Confusión 
from sklearn.metrics import confusion_matrix 
matriz = confusion_matrix(y_test, y_pred) 
print('Matriz de Confusión:') 
print(matriz) 

#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision) 
