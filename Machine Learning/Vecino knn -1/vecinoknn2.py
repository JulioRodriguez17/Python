import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Cargar los datos escalados
df = pd.read_csv('datos_cervezas_escalados.csv')

#Creamos un nuevo escalador
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(df[['ABV', 'IBU', 'EBC', 'Temp_consumo', 'Calificacion', 'Densidad_inicial']])  # Asumiendo los nombres correctos
y = df['Tipo_cerveza']  # Asegúrate de que este es el nombre correcto de la columna de etiquetas

# Crear el clasificador K-NN
knn = KNeighborsClassifier(n_neighbors=15)

# Definir las métricas a utilizar
scoring = ['accuracy', 'recall_macro', 'f1_macro']

# Realizar la validación cruzada
scores = cross_validate(knn, X, y, scoring=scoring,cv=10)

#Entrenamos el clasificador
knn.fit(X,y)

#Escalaremos los datos de la nueva cerveza
nueva_cerveza = pd.DataFrame({
    'ABV': [10],
    'IBU': [117],
    'EBC': [55],
    'Temp_consumo': [3],
    'Calificacion': [4],
    'Densidad_inicial': [1.065]
})
nueva_cerveza_scaled=scaler.transform(nueva_cerveza)

print("Clase",knn.predict(nueva_cerveza_scaled))
print("Probabilidad por clase",knn.predict_proba(nueva_cerveza_scaled))

# Graficacion de los datos
new_df = pd.read_csv('datos_cervezas.csv')
colors = {
    'Lager': 'red',
    'Pale Ale': 'blue',
    'Stout': 'green',
    'IPA': 'orange'
}

# plt.scatter(new_df['Calificacion'], new_df['IBU'], color='blue', label='Datos originales', alpha=0.5)
for class_label, color in colors.items():
    subset = new_df[new_df['Tipo_cerveza'] == class_label]
    plt.scatter(subset['Calificacion'], subset['IBU'], c=color, label=class_label, alpha=0.5)
plt.scatter(nueva_cerveza_scaled[:, 4], nueva_cerveza_scaled[:, 1], color='green', label='Nueva cerveza', marker="P", s=100)

plt.xlabel('Calificación del Usuario')
plt.ylabel('IBU (International Bitterness Units)')
plt.legend(title='Tipo de Cerveza')
plt.grid(True)
plt.show()
