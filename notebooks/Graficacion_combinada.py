import matplotlib.pyplot as plt
import numpy as np

# --- 1. Datos ---
labels = ['1.0E-05', '1.0E-04', '5.0E-04', '1.0E-03', '5.0E-03', '1.0E-02', '5.0E-02']
#mae_sbp = [12.92, 12.54, 10.95, 9.17, 6.35, 7.75, 20.88]
#imp_sbp = [91, 90, 89, 87, 74, 69, 34]
mae_dbp = [6.42, 6.27, 5.71, 5.15, 3.62, 4.14, 18.94]
imp_dbp = [81, 81, 76, 78, 75, 67, 17]
x_pos = np.arange(len(labels))

# --- 2. Configuración Estética ---
color_barras = '#6A4C93'
color_linea = '#009688'
color_texto = 'black'

# Tamaño compacto
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

# --- 3. Gráfico de Barras (Eje Izquierdo) ---
barras = ax1.bar(x_pos, mae_dbp, color=color_barras, alpha=0.85, width=0.4, label='MAE Post Ajuste')

# Configuración Eje Izquierdo
ax1.set_xlabel('Tasa de Aprendizaje', fontsize=10, color=color_texto)
ax1.set_ylabel('MAE Post Ajuste [mmHg]', fontsize=10, color=color_texto)
ax1.tick_params(axis='both', colors=color_texto, labelsize=9)
ax1.set_ylim(0, 25)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=0, fontsize=9, color=color_texto)

# --- 4. Gráfico de Línea (Eje Derecho) ---
ax2 = ax1.twinx()
linea = ax2.plot(x_pos, imp_dbp, color=color_linea, marker='o', 
                 linewidth=2, markersize=7, label='Tasa de Mejora')

# Configuración Eje Derecho (CAMBIO: Sin el [%])
ax2.set_ylabel('Tasa de mejora [%]', fontsize=10, rotation=270, labelpad=15, color=color_texto)
ax2.tick_params(axis='y', colors=color_texto, labelsize=9)
ax2.set_ylim(0, 105)

# --- 5. Detalles Finales ---
# Título (CAMBIO: Título simplificado)
plt.title('Impacto de la Tasa de Aprendizaje para DBP', 
          fontsize=12, pad=15, color=color_texto)

ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax2.set_zorder(ax1.get_zorder() + 1)
ax1.patch.set_visible(False)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor(color_texto)

# Leyenda
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=True, facecolor='white', edgecolor='black', fontsize=9)

plt.tight_layout()
# plt.savefig('grafico_sbp_final.png', bbox_inches='tight') # Descomenta para guardar
plt.show()