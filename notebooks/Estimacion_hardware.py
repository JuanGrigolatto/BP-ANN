import torch
import torch.nn as nn
from src.models.ConvolucionalV1 import Modelo_ConvolucionalV1  # Asegúrate que el import sea correcto según tu estructura

def dimensionar_micro(model, input_size, training_mode="full"):
    """
    training_mode: 
      - 'inference': Solo inferencia (Pesos en Flash).
      - 'full': Entrenar toda la red (Pesos en RAM + Grads + Optimizador).
      - 'last_layer': Entrenar solo la última capa (Head en RAM, resto en Flash).
    """
    
    # --- 1. CAPTURA DE ACTIVACIONES (Fix del Batch=1) ---
    layer_sizes = []
    def hook_fn(module, input, output):
        # Asumimos batch_size=1 para microcontroladores
        out_flat = output.view(-1) 
        layer_sizes.append(out_flat.element_size() * out_flat.nelement())
    
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear, nn.LSTM, nn.GRU)):
            hooks.append(layer.register_forward_hook(hook_fn))
            
    dummy_input = torch.zeros(input_size)
    
    # [FIX CRÍTICO]: Guardamos estado, pasamos a eval() para medir, y restauramos.
    # Esto evita que BatchNorm explote con Batch=1 durante la medición.
    original_mode = model.training
    model.eval()
    
    with torch.no_grad(): # Ahorramos cálculo de gradientes durante la medición
        model(dummy_input)
    
    model.train(original_mode) # Restauramos el estado original
    # -----------------------------------------------------------
    
    # Limpiar hooks
    for h in hooks: h.remove()
    
    # --- CÁLCULOS ---
    
    # Tamaño de parámetros (Pesos y Bias)
    total_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Tamaño de Activaciones (RAM de trabajo)
    # Ping-Pong buffering: Entrada + 2 * Max_Layer_Output
    max_activation_bytes = max(layer_sizes) if layer_sizes else 0
    input_bytes = dummy_input.element_size() * dummy_input.nelement()
    
    # Arena: Espacio dinámico para mover tensores
    arena_bytes = input_bytes + (max_activation_bytes * 2) 

    # --- LÓGICA DE META-LEARNING (PRAGMÁTICA) ---
    
    flash_requirement = 0
    ram_static_requirement = 0 
    ram_dynamic_requirement = arena_bytes 
    
    if training_mode == "inference":
        flash_requirement = total_params_bytes
        ram_static_requirement = 0 # No hay estados mutables
        
    elif training_mode == "full":
        # Escenario: Entrenamiento ONLINE
        flash_requirement = 0 
        
        # 1. Pesos en RAM (para poder editarlos)
        ram_static_requirement = total_params_bytes 
        
        # 2. Gradientes (mismo tamaño que pesos)
        ram_static_requirement += total_params_bytes 
        
        # 3. Estado del Optimizador (IMPORTANTE)
        # Si usas SGD simple: 0 bytes extra (o muy poco).
        # Si usas SGD con Momentum: +1 copia de pesos.
        # Si usas Adam: +2 copias de pesos (mean, variance).
        # ASUMIMOS SGD SIMPLE PARA MICROCONTROLADOR (Escenario optimista):
        optimizer_overhead = 0 
        ram_static_requirement += optimizer_overhead
        
    elif training_mode == "last_layer":
        # Buscamos params de la última capa (Classifier)
        # Asumiendo que la última capa es lineal y está al final de modules()
        # NOTA: Esto puede fallar si tu última capa no es la última en la lista, 
        # pero para modelos secuenciales simples funciona.
        last_layer = list(model.modules())[-1]
        last_layer_params = sum(p.numel() * p.element_size() for p in last_layer.parameters())
        
        frozen_params = total_params_bytes - last_layer_params
        
        flash_requirement = frozen_params
        # RAM: Pesos LastLayer + Grads LastLayer + Optimizador LastLayer
        ram_static_requirement = last_layer_params * 2 
        
    # Salida
    print(f"--- Dimensionamiento para: {training_mode.upper()} ---")
    print(f"FLASH necesaria (Storage):      {flash_requirement / 1024:.2f} KB")
    print(f"RAM Estática (Pesos+Grads):     {ram_static_requirement / 1024:.2f} KB")
    print(f"RAM Dinámica (Arena/Buffers):   {ram_dynamic_requirement / 1024:.2f} KB")
    print(f"------------------------------------------------")
    print(f"TOTAL RAM MÍNIMA (Estimada):    {(ram_static_requirement + ram_dynamic_requirement) / 1024:.2f} KB")
    print("------------------------------------------------")

if __name__ == '__main__':
    
    IN_CHANNELS = 2
    OUT_CHANNELS = 2
    LARGO_SENAL = 500

    model = Modelo_ConvolucionalV1(
        in_channels=IN_CHANNELS, 
        out_channels=OUT_CHANNELS, 
        long_signal=LARGO_SENAL
    )

    input_size = (1, IN_CHANNELS, LARGO_SENAL)

    print("-" * 80)
    print(f"REPORTE DE DIMENSIONAMIENTO DE HARDWARE (TRL 4) - MODELO ConvolucionalV1")
    print(f"Configuración: {IN_CHANNELS} in -> {OUT_CHANNELS} out | Largo Señal: {LARGO_SENAL}")
    print("-" * 80)

    # Escenario 1: Solo correr el modelo
    dimensionar_micro(model, input_size, training_mode="inference")
    
    # Escenario 2: Meta-Learning completo en el chip
    dimensionar_micro(model, input_size, training_mode="full")
    
    # Escenario 3: Transfer Learning ligero (solo ajustar la salida)
    # dimensionar_micro(model, input_size, training_mode="last_layer")