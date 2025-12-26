import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo
df = pd.read_csv('graficas/training_log.csv')

# 2. Configuración del gráfico
plt.figure(figsize=(10, 6)) 
plt.style.use('seaborn-v0_8-whitegrid') 

# 3. Graficar las líneas
# 'train_loss'
plt.plot(df['epoch'], df['train_loss'], 
         label='Entrenamiento', 
         color='#1f77b4')

# 'valid_loss'
plt.plot(df['epoch'], df['valid_loss'], 
         label='Validación', 
         color='#ff7f0e')

# 4. Etiquetas y Títulos
plt.xlabel('Epoca')
plt.ylabel('Pérdida (Loss)')
plt.legend() 
# 5. Ajustes finales
plt.xlim(df['epoch'].min(), df['epoch'].max()) 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 6. Mostrar y Guardar
plt.tight_layout() 
plt.savefig('curva_loss.png', dpi=300) 
plt.show() 