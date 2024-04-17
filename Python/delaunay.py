import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Generamos un conjunto de puntos aleatorios
np.random.seed(0)  # Para reproducibilidad
points = np.random.rand(30, 2)  # 30 puntos en 2D

# Realizamos la triangulación de Delaunay
tri = Delaunay(points)

# Visualizamos la triangulación
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')  # Marcamos los puntos

plt.show()