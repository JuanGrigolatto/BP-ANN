import torch
import torch.nn as nn
import sys
import os

# Ajuste de path para encontrar src/
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1
    print("[INFO] Modelo_ConvolucionalV1 cargado correctamente.")
except ImportError:
    print("[ERROR] No se encontró el modelo. Verifica la ruta.")
    sys.exit()

def analizar_V1_final(model, input_size):
    total_macs = 0
    total_params = sum(p.numel() for p in model.parameters())
    layer_sizes = []

    # --- HOOK PARA ACTIVACIONES Y MACS ---
    def hook_fn(module, input, output):
        nonlocal total_macs
        if isinstance(output, torch.Tensor):
            out_flat = output.view(-1)
            layer_sizes.append(out_flat.element_size() * out_flat.nelement())
        
        # Cálculo de MACs según tipo de capa
        if isinstance(module, nn.Conv1d):
            # MACs = Out_Ch * In_Ch * Kernel * Out_Len
            batch, in_ch, length = input[0].size()
            out_ch, _, kernel = module.weight.size()
            out_len = output.size(-1)
            total_macs += out_ch * in_ch * kernel * out_len
            
        elif isinstance(module, nn.Linear):
            # MACs = In_Features * Out_Features
            total_macs += module.in_features * module.out_features

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # --- EJECUCIÓN (BATCH SIZE = 1 para Embebidos) ---
    dummy_input = torch.zeros(input_size)
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    for h in hooks: h.remove()

    # --- CÁLCULOS TRL 4 (ESCENARIO ADAM) ---
    byte_size = 4 # Float32
    pesos_bytes = total_params * byte_size
    grads_bytes = pesos_bytes
    adam_bytes = 2 * pesos_bytes # m y v
    system_bytes = 48 * 1024 # RTOS + ADC Buffers
    max_act = max(layer_sizes) if layer_sizes else 0
    arena_bytes = (dummy_input.element_size() * dummy_input.nelement()) + (max_act * 2)
    
    total_ram_kb = (pesos_bytes + grads_bytes + adam_bytes + arena_bytes + system_bytes) / 1024

    # --- ESTIMACIÓN DE DESEMPEÑO ---
    # Frecuencia típica de un micro de alta gama (STM32H7 o i.MX RT)
    mhz_objetivo = 480 
    # Un Cortex-M7 con instrucciones DSP hace aprox 1 MAC por ciclo
    latencia_inf_ms = (total_macs / (mhz_objetivo * 1e6)) * 1000
    # Escenario Fine-Tuning (MAML): Forward + Backward + Update (Adam)
    # Se estima un factor x3 para el ciclo completo de entrenamiento por muestra
    latencia_train_ms = latencia_inf_ms * 3

    print("\n" + "="*70)
    print(f" REPORTE TÉCNICO V1: 2 CANALES X 500 MUESTRAS (TRL 4)")
    print("="*70)
    print(f"Parámetros Totales:      {total_params:,}")
    print(f"Cómputo Total (MACs):    {total_macs:,} operaciones")
    print(f"Densidad de Cómputo:     {total_macs / 1e6:.2f} MMACs")
    print("-" * 70)
    print(f"MEMORIA RAM (Total):     {total_ram_kb:>8.2f} KB ({total_ram_kb/1024:.2f} MB)")
    print(f"MEMORIA FLASH (Mínima):  {(pesos_bytes / 1024) + 128:>8.2f} KB")
    print("-" * 70)
    print(f"ESTIMACIÓN DE TIEMPOS (@ {mhz_objetivo} MHz):")
    print(f"  [>] Inferencia (1 paso):        {latencia_inf_ms:.2f} ms")
    print(f"  [>] Ajuste Fino (Adam Step):    {latencia_train_ms:.2f} ms")
    print("=" * 70)

    # Diagnóstico Final
    if total_ram_kb > 1024:
        print("DIAGNÓSTICO: Requiere SDRAM/PSRAM externa (>1MB RAM).")
    if latencia_train_ms > 100:
        print("AVISO: Latencia de entrenamiento alta. Evaluar optimización por hardware.")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    # Configuración según tu requerimiento: 2 canales, 500 muestras
    IN_CH = 2
    OUT_CH = 2
    L = 500 
    
    v1_model = Modelo_ConvolucionalV1(in_channels=IN_CH, out_channels=OUT_CH, long_signal=L)
    
    # Input size: (Batch, Channels, Length)
    input_size = (1, IN_CH, L)
    analizar_V1_final(v1_model, input_size)