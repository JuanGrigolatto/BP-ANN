"""
Módulo: Graficacion_perdida.py
Autor: Juan Marcos Grigolatto
Descripción: Script para la generación de gráficas de Meta-Entrenamiento (MAML). Lee los archivos de log (.csv) 
             y plotea las curvas de Meta-Pérdida (Meta-Loss) para los conjuntos 
             de Entrenamiento y Validación, permitiendo evaluar el aprendizaje 
             de la inicialización óptima y detectar posible sobreajuste.
"""
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo
df = pd.read_csv('metalearning/logs/log_STAGE2_DELTA_Specialist_MSL.csv')

# 2. Configuración del gráfico
plt.figure(figsize=(10, 6)) 

# 3. Graficar las líneas
plt.plot(df['epoch'], df['train_loss'], 
         label='Meta Pérdida Entrenamiento',  
         color='#1f77b4',      
         marker='.',            
         linestyle='-',         
         linewidth=1,           
         alpha=0.8)             

# Validación: En la imagen es naranja con puntos grandes y sólidos
plt.plot(df['epoch'], df['valid_loss'], 
         label='Meta Pérdida Validación', 
         color='#ff7f0e',       
         marker='o',            
         linestyle='-',
         linewidth=2,          
         markersize=6)         

# 4. Etiquetas y Títulos
plt.xlabel('Época')             
plt.ylabel('Pérdida')     

plt.legend(loc='upper right', frameon=True) 

# 5. Ajustes finales 
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# 6. Mostrar y Guardar
plt.tight_layout() 
plt.savefig('metalearning/logs/curva_loss_espanol.png', dpi=300) 
plt.show()