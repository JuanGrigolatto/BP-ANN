import matplotlib.pyplot as plt
import numpy as np

# --- 1. Datos DBP (Diastólica) ---
# Datos actualizados según el grid search
labels = ['1.0E-05', '1.0E-04', '5.0E-04', '1.0E-03', '5.0E-03', '1.0E-02', '5.0E-02']
rmse_dbp = [12.97, 12.70, 11.57, 10.35, 6.80, 8.04, 32.08] # Datos DBP
imp_dbp = [87, 87, 88, 85, 71, 67, 22]                    # Datos DBP
x_pos = np.arange(len(labels))

# --- 2. Configuración Estética (Colores Pastel) ---
# Violeta Pastel (Para las barras de error)
color_barras = '#B39DDB'  # RGB: (179, 157, 219)
# Verde Menta Pastel (Para la línea de mejora - un tono sólido para que se vea sobre blanco)
color_linea = '#4DB6AC'   # RGB: (77, 182, 172)
color_texto = '#37474F'   # Gris oscuro profesional (Blue Grey 800)

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

# --- 3. Gráfico de Barras (Eje Izquierdo - RMSE) ---
# Zorder bajo para que quede detrás
barras = ax1.bar(x_pos, rmse_dbp, color=color_barras, alpha=0.9, width=0.5, 
                 label='RMSE Post Ajuste', zorder=1)

# Valores sobre las barras clave (Mínimo en index 4 y Siguiente en index 5)
ax1.text(x_pos[4], rmse_dbp[4] + 0.5, f'{rmse_dbp[4]:.1f}', ha='center', va='bottom', 
         fontsize=9, fontweight='bold', color=color_texto)
ax1.text(x_pos[5], rmse_dbp[5] + 0.5, f'{rmse_dbp[5]:.1f}', ha='center', va='bottom', 
         fontsize=9, color=color_texto)

# Configuración Eje Izquierdo
ax1.set_xlabel('Tasa de Aprendizaje (Learning Rate)', fontsize=10, color=color_texto)
# CAMBIO: Etiqueta DBP
ax1.set_ylabel('RMSE Global DBP [mmHg]', fontsize=10, color=color_texto)
ax1.tick_params(axis='both', colors=color_texto, labelsize=9)

# CAMBIO: Límite aumentado a 40 para acomodar el valor de 32.08
ax1.set_ylim(0, 40)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=0, fontsize=9, color=color_texto)

# --- 4. Gráfico de Línea (Eje Derecho - Tasa de Mejora) ---
ax2 = ax1.twinx()
linea = ax2.plot(x_pos, imp_dbp, color=color_linea, marker='o', 
                 linewidth=3, markersize=8, label='Tasa de Mejora', zorder=2)

# --- FLECHA CORREGIDA ---
# CAMBIO: Apunta al índice 4 (5.0E-03) donde ocurre la caída fuerte en DBP (de 85% a 71%)
ax2.annotate('Caída de estabilidad', 
             xy=(x_pos[4], imp_dbp[4]),         # Punta de la flecha (en el punto 71%)
             xytext=(x_pos[4], imp_dbp[4]+15),  # Texto más arriba
             arrowprops=dict(facecolor=color_texto, arrowstyle='->', lw=1.5), 
             fontsize=9, ha='center', color=color_texto)

# Configuración Eje Derecho
ax2.set_ylabel('Pacientes con Mejora [%]', fontsize=10, rotation=270, labelpad=20, color=color_texto)
ax2.tick_params(axis='y', colors=color_texto, labelsize=9)

# CAMBIO: Rango ampliado (10 a 115) para que entre el valor bajo de 22%
ax2.set_ylim(10, 115)

# --- 5. Detalles Finales ---
# CAMBIO: Título DBP
plt.title('Impacto de la Tasa de Aprendizaje en DBP: Precisión vs Estabilidad', 
          fontsize=12, pad=15, color=color_texto, fontweight='medium')

# Grid estética
ax1.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

# Quitar marco superior
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Colorear bordes restantes
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor(color_texto)

# Leyenda unificada
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=2, frameon=False, fontsize=9, labelcolor=color_texto)

plt.tight_layout()
plt.show()