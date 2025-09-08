BP-ANN-clean/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── datos/
│   │       ├── Part_1.mat
│   │       ├── Part_2.mat
│   │       └── ... 
│   ├── interim/
│   │   └── (archivos temporales, por ejemplo: patients_temp.dat)
│   └── processed/
│       ├── Errores_predicción.npy
│       ├── Errores_predicción.npz
│       ├── indices_errores.npy
│       └── data_UCI/
│           ├── dataset_parte_1_data.dat
│           └── ...
│
├── notebooks/
│   ├── Analisis_error.py
│   ├── Prueba_recorte.py
│   ├── prueba_seniales_ruido.py
│   └── (otros scripts de análisis/visualización)
│
├── src/
│   ├── features/
│   │   ├── data_processing/
│   │   │   ├── Procesamiento_datos_UCI.py
│   │   │   ├── Procesamiento_por_picos.py
│   │   │   └── Procesamiento_ventana_fija.py
│   │   └── Detector_de_picos.py
│   ├── data/
│   │   ├── data_chargers/
│   │   │   ├── Clase_UCIDataset.py
│   │   │   ├── MetaDataset.py
│   │   │   └── Tuningndataset.py
│   ├── models/
│   │   └── (código de modelos)
│   ├── utils/
│   │   ├── Tools/
│   │   └── Visualizador_datos.py
│   └── entrenamiento/
│       └── (scripts de entrenamiento)
│
├── models/
│   ├── best_models/
│   │   ├── best_model.pt
│   │   ├── best_meta_model.pt
│   │   └── ...
│   ├── tuning_model.pt
│   └── (otros modelos guardados)
│
├── tests/
│   └── (tests unitarios y de integración)
│
├── graficas/
│   └── (graficas varias)
│
└── metalearning/
    └── (scripts y datos relacionados a meta-learning)