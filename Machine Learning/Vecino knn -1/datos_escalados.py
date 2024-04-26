from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd

# Cargar datos
df = pd.read_csv('datos_cervezas.csv')

# Seleccionar las características numéricas para escalar
features = df[['ABV', 'IBU', 'EBC', 'Temp_consumo', 'Calificacion', 'Densidad_inicial']]

# Crear el escalador
scaler = preprocessing.MinMaxScaler()

# Ajustar y transformar los datos
scaled_features = scaler.fit_transform(features)

# Crear un nuevo DataFrame con las características escaladas
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Agregar la columna de etiquetas al DataFrame escalado
scaled_df['Tipo_cerveza'] = df['Tipo_cerveza']

# Opcionalmente, guardar el nuevo DataFrame escalado en un CSV
scaled_df.to_csv('datos_cervezas_escalados.csv', index=False)

print(scaled_df.head())  # Imprimir las primeras filas para ver el resultado
