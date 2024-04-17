import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Generamos puntos aleatorios que representarán las torres de telefonía celular
np.random.seed(0)  # Para reproducibilidad
points = np.random.rand(10, 2)  # 10 puntos en 2D

# Generamos el diagrama de Voronoi para estos puntos
vor = Voronoi(points)

# Visualizamos el diagrama de Voronoi
fig, ax = plt.subplots()
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=10)

# Dibujamos los puntos (torres de telefonía celular)
ax.plot(points[:,0], points[:,1], 'b.', markersize=10)

plt.show()