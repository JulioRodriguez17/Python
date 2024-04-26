import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('datos_cervezas.csv')

# Crear un diccionario de colores para las clases
colors = {
    'Lager': 'red',
    'Pale Ale': 'blue',
    'Stout': 'green',
    'IPA': 'orange'
}

# Crear una gráfica de dispersión
plt.figure(figsize=(10, 6))
for class_label, color in colors.items():
    subset = df[df['Tipo_cerveza'] == class_label]
    plt.scatter(subset['Calificacion'], subset['IBU'], c=color, label=class_label, alpha=0.5)

plt.title('Dispersión de Calificacion vs IBU por Tipo de Cerveza')
plt.xlabel('Calificación del Usuario')
plt.ylabel('IBU (International Bitterness Units)')
plt.legend(title='Tipo de Cerveza')
plt.grid(True)
plt.show()

# Imprimir la cantidad de objetos etiquetados con cada clase
print("Cantidad de objetos por clase:")
print(df['Tipo_cerveza'].value_counts())