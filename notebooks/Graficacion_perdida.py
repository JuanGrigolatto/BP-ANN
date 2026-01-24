import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo
# Asegúrate de que las columnas 'epoch', 'train_loss' y 'valid_loss' existan en tu CSV
df = pd.read_csv('metalearning/logs/training_log.csv')

# 2. Configuración del gráfico
plt.figure(figsize=(10, 6)) 

# 3. Graficar las líneas
# Entrenamiento: En la imagen es azul con puntos pequeños
plt.plot(df['epoch'], df['train_loss'], 
         label='Meta Pérdida Entrenamiento',  # Ajustado al contexto "Meta" de la imagen
         color='#1f77b4',       # Azul estándar
         marker='.',            # Punto pequeño
         linestyle='-',         # Línea sólida
         linewidth=1,           # Grosor fino para no tapar los datos
         alpha=0.8)             # Ligera transparencia

# Validación: En la imagen es naranja con puntos grandes y sólidos
plt.plot(df['epoch'], df['valid_loss'], 
         label='Meta Pérdida Validación', 
         color='#ff7f0e',       # Naranja estándar
         marker='o',            # Círculo grande (como en la imagen)
         linestyle='-',
         linewidth=2,           # Línea un poco más gruesa para resaltar validación
         markersize=6)          # Tamaño del marcador visible

# 4. Etiquetas y Títulos
plt.xlabel('Época')             # Tilde agregada
plt.ylabel('Pérdida (MSE)')     # Asumo MSE porque la imagen dice "Loss (MSE)"

# Leyenda
plt.legend(loc='upper right', frameon=True) # Ubicación y recuadro como en la imagen

# 5. Ajustes finales (Críticos para replicar el estilo)
# La grilla es fundamental para leer valores exactos, la descomenté y estilicé
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Opcional: Si quieres que el eje X sea solo enteros (ya que son épocas)
# from matplotlib.ticker import MaxNLocator
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# 6. Mostrar y Guardar
plt.tight_layout() 
plt.savefig('metalearning/logs/curva_loss_espanol.png', dpi=300) 
plt.show()