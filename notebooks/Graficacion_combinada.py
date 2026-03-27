import matplotlib.pyplot as plt
import numpy as np
"""
Visualización de Resultados de Búsqueda de Hiperparámetros (Few-Shot Learning)
------------------------------------------------------------------------------
Este script genera un gráfico de doble eje para analizar el impacto de la 
tasa de aprendizaje (Learning Rate) en la adaptación intra-paciente de la 
Presión Arterial Diastólica (DBP).

Eje Izquierdo (Barras): Muestra el RMSE global post-ajuste. Evalúa la precisión.
Eje Derecho (Línea): Muestra la Tasa de Mejora (%). Evalúa la estabilidad poblacional.

Objetivo: Justificar metodológicamente la elección del hiperparámetro óptimo 
evidenciando el punto de quiebre donde una tasa muy alta genera sobreajuste 
(overfitting) en el conjunto de soporte, reduciendo el error promedio pero 
perjudicando la estimación en la mayoría de los pacientes individuales.
"""
variable_a_graficar = 'DBP' 

labels = ['1.0E-05', '1.0E-04', '5.0E-04', '1.0E-03', '5.0E-03', '1.0E-02', '5.0E-02']

datos = {
    'SBP': {
        'rmse': [20.55, 20.35, 19.45, 18.38, 12.73, 13.35, 17.95],
        'imp':  [91, 91, 90, 88, 83, 76, 52],
        'ylim_rmse': (0, 25),   # Límite dinámico del eje Y para SBP
        'ylim_imp': (40, 105),  # Límite dinámico para la mejora en SBP
        'offset_flecha': 12
    },
    'DBP': {
        'rmse': [14.48, 14.31, 13.60, 12.80, 8.83, 7.93, 14.12],
        'imp':  [92, 90, 88, 83, 74, 67, 35],
        'ylim_rmse': (0, 20),
        'ylim_imp': (20, 105),
        'offset_flecha': 12
    }
}

rmse_actual = datos[variable_a_graficar]['rmse']
imp_actual = datos[variable_a_graficar]['imp']
ylim_rmse = datos[variable_a_graficar]['ylim_rmse']
ylim_imp = datos[variable_a_graficar]['ylim_imp']
offset = datos[variable_a_graficar]['offset_flecha']

x_pos = np.arange(len(labels))

color_barras = '#B39DDB'  
color_linea = '#4DB6AC'   
color_texto = '#37474F'   

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

barras = ax1.bar(x_pos, rmse_actual, color=color_barras, alpha=0.9, width=0.5, 
                 label='RMSE Post Ajuste', zorder=1)

ax1.text(x_pos[4], rmse_actual[4] + 0.3, f'{rmse_actual[4]:.2f}', ha='center', va='bottom', 
         fontsize=9, fontweight='bold', color=color_texto)

ax1.set_xlabel('Tasa de Aprendizaje (Learning Rate)', fontsize=10, color=color_texto)
ax1.set_ylabel(f'RMSE Global {variable_a_graficar} [mmHg]', fontsize=10, color=color_texto)
ax1.tick_params(axis='both', colors=color_texto, labelsize=9)

ax1.set_ylim(ylim_rmse)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=0, fontsize=9, color=color_texto)

ax2 = ax1.twinx()
linea = ax2.plot(x_pos, imp_actual, color=color_linea, marker='o', 
                 linewidth=3, markersize=8, label='Tasa de Mejora', zorder=2)

# --- FLECHA (Apunta a la caída de estabilidad en 5.0E-03) ---
ax2.annotate('Caída de estabilidad', 
             xy=(x_pos[4], imp_actual[4]),         
             xytext=(x_pos[4], imp_actual[4] + offset),  
             arrowprops=dict(facecolor=color_texto, arrowstyle='->', lw=1.5), 
             fontsize=9, ha='center', color=color_texto)

ax2.set_ylabel('Pacientes con Mejora [%]', fontsize=10, rotation=270, labelpad=20, color=color_texto)
ax2.tick_params(axis='y', colors=color_texto, labelsize=9)
ax2.set_ylim(ylim_imp)

plt.title(f'Impacto de la Tasa de Aprendizaje en {variable_a_graficar}: Precisión vs Estabilidad', 
          fontsize=12, pad=15, color=color_texto, fontweight='medium')

ax1.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor(color_texto)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=2, frameon=False, fontsize=9, labelcolor=color_texto)

plt.tight_layout()

# plt.savefig(f'Hyperparam_Tuning_{variable_a_graficar}.png', dpi=300, bbox_inches='tight')

plt.show()