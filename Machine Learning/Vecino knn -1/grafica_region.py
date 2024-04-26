import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Suponiendo que df es tu DataFrame y que ya ha sido cargado y preparado
df = pd.read_csv('datos_cervezas.csv')

# Aquí, seleccionamos solo dos características para la visualización
features = ['Calificacion', 'IBU']
X = df[features].values
y = df['Tipo_cerveza'].astype('category').cat.codes.values  # Convertimos etiquetas a códigos numéricos

# Escalar características para mejorar la visualización
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Crear el clasificador K-NN
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Paso para crear la malla: definir los márgenes de la malla y crear una malla de puntos
h = .02  # tamaño de paso en la malla
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Predecir la clase para cada punto en la malla
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Utilizar el resultado para mostrar las regiones de decisión
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)

# También graficar los puntos de datos
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k', s=20)
plt.xlabel('Calificación del Usuario')
plt.ylabel('IBU (escalado)')
plt.title("Region de las 4 clases de cerveza")
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()
