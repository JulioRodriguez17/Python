import numpy as np
import pandas as pd

#Establecemos la semilla
np.random.seed(0)

#Definimos nuestro numero de registros
n_registros = 200

#Generamos los datos para las caracteristicas, enfocandonos en cervezas
abv = np.random.uniform(3, 12, n_registros)  # ABV(Alcohol by volumen) de 3% a 12%
ibu = np.random.uniform(1, 120, n_registros)  # IBU unidad de amargura de la cerveza que va de 1 a 120
ebc = np.random.uniform(8, 80, n_registros)  # EBC Color de la cerveza que indica que tan oscura es y va de 8 a 80 (pálido a oscuro)
temp_consumo = np.random.uniform(3, 12, n_registros)  # Temperatura de consumo de 3°C a 12°C
calificacion = np.random.uniform(1, 5, n_registros)  # Calificación de los usuarios que va de 1 a 5
densidad_inicial = np.random.uniform(1.030, 1.080, n_registros)  # Densidad del mosto antes de la fermentación, que puede indicar el potencial de alcohol y cuerpo de la cerveza.

#Creamos las etiquetas de las cervezas
    # Lager: Cervezas fermentadas a baja temperatura, generalmente más ligeras y refrescantes.
    # Ale: Cervezas fermentadas a temperaturas más altas, con sabores más frutales y complejos.
    # Stout: Cervezas muy oscuras y a menudo cremosas con sabores robustos de malta tostada.
    # IPA (India Pale Ale): Conocidas por su fuerte amargura y sabores/aromas florales o cítricos.
label_cervezas= np.random.choice(['Lager', 'Pale ALe', 'Stout', 'IPA'], n_registros)

#Creamos un DataFrame con los datos
df_cervezas = pd.DataFrame({
    'ABV': abv,
    'IBU': ibu,
    'EBC': ebc,
    'Temp_consumo': temp_consumo,
    'Calificacion': calificacion,
    'Densidad_inicial': densidad_inicial,
    'Tipo_cerveza': label_cervezas
})

#Guardamos los datos en un archivo csv
df_cervezas.to_csv('datos_cervezas.csv', index=False)