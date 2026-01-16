import torch
from torchinfo import summary
import sys
import os

# Agregamos el path para que encuentre tus módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Importamos tu modelo V1
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
from src.models.ConvolucionalV2 import Modelo_ConvolucionalV2
from src.models.InceptionTime import InceptionTime

def main():
    # --- CONFIGURACIÓN  ---
    IN_CHANNELS = 2     # ECG, PPG
    OUT_CHANNELS = 2    # SBP, DBP 
    
    LARGO_SENAL = 500   
    
    # -----------------------------------------------------------

   
    # --- INICIALIZACIÓN DEL MODELO ---
    """
    model = Modelo_ConvolucionalV1(
        in_channels=IN_CHANNELS, 
        out_channels=OUT_CHANNELS, 
        long_signal=LARGO_SENAL
    )
    """
    model=InceptionTime(
        c_in=2, 
        c_out=2, 
        seq_len=None,
        n_filters=32,
        depth = 6
    )

    input_size = (1, IN_CHANNELS, LARGO_SENAL)

    print("-" * 80)
    print(f"REPORTE DE DIMENSIONAMIENTO DE HARDWARE (TRL 4) - MODELO TIME 32")
    print(f"Configuración: {IN_CHANNELS} in -> {OUT_CHANNELS} out | Largo Señal: {LARGO_SENAL}")
    print("-" * 80)

  
    try:
        model_stats = summary(
            model, 
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            col_width=20,
            row_settings=["var_names"],
            verbose=1
        )
        
        flash_size = model_stats.total_param_bytes / (1024 ** 2)
        
        ram_bytes = model_stats.total_output_bytes + model_stats.total_param_bytes
        ram_mb = (ram_bytes / (1024 ** 2)) 

        # Operaciones (FLOPs)
        flops_m = model_stats.total_mult_adds / 1e6

        print("\n" + "="*40)
        print(" ANÁLISIS DE VIABILIDAD (MODELO TIME 32)")
        print("="*40)
        print(f"1. ALMACENAMIENTO (Flash):  {flash_size:.2f} MB")
        print(f"   (Peso del archivo .pt cuantizado o en crudo)")
        
        print(f"2. MEMORIA RAM MÍNIMA:    {ram_mb:.2f} MB")
        print(f"   (Memoria dinámica necesaria para correr)")
        
        print(f"3. COMPUTACIÓN:           {flops_m:.2f} MFLOPs")
        print("="*40)
        
       
        if flash_size < 1.0 and ram_mb < 0.25:
            print(">> DIAGNÓSTICO: Apto para Microcontroladores gama media (STM32F4/L4).")
        elif flash_size < 2.0 and ram_mb < 0.5:
            print(">> DIAGNÓSTICO: Requiere Microcontrolador gama alta (STM32H7, Teensy 4.1).")
        else:
            print(">> DIAGNÓSTICO: Modelo PESADO. Recomendado Single Board Computer (Raspberry Pi / Jetson).")

    except Exception as e:
        print(f"\nError al analizar el modelo: {e}")
        print("Verifica que 'long_signal' coincida con la estructura interna de tus capas Linear.")

if __name__ == '__main__':
    main()