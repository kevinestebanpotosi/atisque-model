# Preparación de Datos para KPU Depth Estimation

Esta guía explica cómo preparar y organizar los datos para el entrenamiento y compilación del modelo.

## 📥 Dataset NYU Depth V2

### Descripción
El NYU Depth V2 es un dataset de imágenes RGB-D de escenas de interiores, capturado con Microsoft Kinect. Contiene:
- **47,584 imágenes de entrenamiento**
- **654 imágenes de validación**
- **Pares RGB-Depth alineados**
- **Escenas de 464 ambientes diferentes**

### Estructura Original
Los datos están organizados en archivos HDF5 (.h5) comprimidos en archivos .tar:

```
Original Structure:
train-000000.tar
├── nyu_depth_v2_labeled.mat
├── train/
│   ├── 00001.h5
│   ├── 00002.h5
│   └── ...
train-000001.tar
└── ...
```

## 🗂️ Organización para Este Proyecto

### Estructura Requerida
```
data/
├── train-000000.tar
├── train-000001.tar
├── ...
├── train-000011.tar    (12 archivos en total)
├── val-000000.tar
└── val-000001.tar      (2 archivos en total)
```

### Para Calibración (Cuantización)
```
data/
├── calibration/        # Para compilación cuantizada
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...            # 100-1000 imágenes representativas
└── test_images/       # Para validación
    ├── test1.jpg
    ├── test2.jpg
    └── ...
```

## 🔄 Proceso de Descarga

### Opción 1: Descarga Manual (Recomendada)
1. Visitar: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Descargar los archivos:
   - `Labeled dataset (train)` (~2.8GB)
   - `Labeled dataset (val)` (~2.8GB)
3. Extraer y organizar en la estructura anterior

### Opción 2: Usando Scripts (Si disponibles)
```bash
# Ejemplo de script de descarga
python scripts/download_dataset.py \
  --output-dir data \
  --split train val
```

### Opción 3: Hugging Face Datasets (Requiere ~35GB)
```python
from datasets import load_dataset

# Cargar dataset completo
dataset = load_dataset('sayakpaul/nyu_depth_v2')

# Convertir a archivos .tar locales
# (Requiere implementación personalizada)
```

## 🛠️ Preprocesamiento

### Transformaciones Aplicadas en `train_kpu_depth.py`

```python
def transform_batch(examples):
    images = examples['image']
    depths = examples['depth_map']
    
    p_images, p_depths = [], []
    
    for img, depth in zip(images, depths):
        # 1. Resize a 224x224 (compatibilidad con arquitectura)
        img = img.resize((224, 224))
        depth = depth.resize((224, 224))
        
        # 2. Conversión a tensor
        img_t = TF.to_tensor(img)
        
        # 3. Procesamiento de depth map
        depth_arr = np.array(depth, dtype=np.float32)
        depth_t = torch.from_numpy(depth_arr).unsqueeze(0)
        
        # 4. Normalización crítica para KPU: [0, 6]
        d_max = depth_t.max()
        if d_max > 0:
            depth_t = (depth_t / d_max) * 6.0
            
        p_images.append(img_t)
        p_depths.append(depth_t)
        
    return {'pixel_values': p_images, 'labels': p_depths}
```

### Consideraciones Especiales para KPU

1. **Rango [0, 6]**: Las activaciones ReLU6 limitan el output
2. **Forma 224x224**: Arquitectura optimizada para esta resolución
3. **Batch estático**: Siempre batch size 1 para inferencia
4. **Normalización simple**: Solo escalado, sin mean/std complex

## 📊 Estadísticas del Dataset

### Distribución Original
| Split | Imágenes | Tamaño | Resolución Original |
|-------|----------|--------|---------------------|
| Train | 47,584 | ~28GB | 640×480 |
| Val   | 654     | ~0.3GB | 640×480 |

### Después de Preprocesamiento
| Característica | Valor | Notas |
|----------------|-------|-------|
| Resolución | 224×224 | Reducción 8.5× en píxeles |
| Canales RGB | 3 | Mantenido |
| Canales Depth | 1 | Escala de grises |
| Rango Depth | [0, 6] | Normalizado para ReLU6 |
| Tipo de datos | float32 | Para entrenamiento |

## 🧪 Dataset de Calibración

### Propósito
Para compilación cuantizada, se necesita un dataset pequeño para:
- Calcular rangos de activación
- Ajustar parámetros de cuantización
- Minimizar pérdida de precisión

### Creación del Dataset
```python
import os
from PIL import Image
import numpy as np

def create_calibration_dataset(source_dir, output_dir, num_images=500):
    """Crear dataset de calibración a partir del dataset principal."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar dataset
    dataset = load_dataset('./nyu_depth_v2.py', split='train')
    
    # Seleccionar imágenes aleatorias
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        example = dataset[idx]
        image = example['image']
        
        # Aplicar mismo preprocesamiento que en entrenamiento
        image = image.resize((224, 224))
        
        # Guardar como JPEG
        output_path = os.path.join(output_dir, f'calib_{i:04d}.jpg')
        image.save(output_path, 'JPEG', quality=95)
        
        if (i + 1) % 100 == 0:
            print(f"Procesadas {i + 1}/{num_images} imágenes")
    
    print(f"Dataset de calibración creado en {output_dir}")
```

### Características Recomendadas
- **Tamaño**: 100-1000 imágenes
- **Diversidad**: Varias escenas y condiciones de iluminación
- **Formato**: JPEG para reducir espacio
- **Resolución**: 224×224 (igual que inferencia)

## ⚠️ Consideraciones de Memoria

### Limitaciones de KPU K210
- **Memoria total**: ~6MB
- **Memoria para modelo**: ~2-3MB
- **Memoria para activaciones**: ~1-2MB
- **Memoria para input/output**: ~1MB

### Implicaciones para Datos
1. **Batch size 1**: Inferencia individual
2. **Resolución limitada**: 224×224 máximo
3. **Cuantización necesaria**: Para modelos grandes
4. **Canales limitados**: Evitar arquitecturas muy anchas

## 🔍 Validación de Calidad de Datos

### Script de Verificación
```python
def validate_dataset(data_dir):
    """Validar que el dataset está correctamente preparado."""
    
    print("Validando dataset...")
    
    # Verificar archivos .tar
    tar_files = list(Path(data_dir).glob("*.tar"))
    if not tar_files:
        print("❌ No se encontraron archivos .tar")
        return False
    
    expected_files = [
        f"train-{i:06d}.tar" for i in range(12)
    ] + [
        f"val-{i:06d}.tar" for i in range(2)
    ]
    
    missing = [f for f in expected_files 
               if not (Path(data_dir) / f).exists()]
    
    if missing:
        print(f"❌ Archivos faltantes: {missing}")
        return False
    
    print(f"✅ {len(tar_files)} archivos .tar encontrados")
    return True
```

### Checks Recomendados
- [ ] Archivos .tar presentes
- [ ] Estructura HDF5 válida
- [ ] Imágenes y depth maps alineados
- [ ] Sin imágenes corruptas
- [ ] Resolución consistente

## 🚀 Flujo de Trabajo Completo

### Paso 1: Descarga
```bash
# Descargar dataset NYU Depth V2
# Manualmente desde el sitio web oficial
```

### Paso 2: Organización
```bash
# Colocar archivos en data/
mkdir -p data
mv *.tar data/
```

### Paso 3: Verificación
```bash
# Verificar estructura
python -c "from pathlib import Path; \
           files = list(Path('data').glob('*.tar')); \
           print(f'{len(files)} archivos encontrados')"
```

### Paso 4: Creación de Dataset de Calibración (Opcional)
```bash
# Solo si planeas cuantización
python scripts/create_calibration.py \
  --source data \
  --output data/calibration \
  --num-images 500
```

### Paso 5: Entrenamiento
```bash
# Iniciar entrenamiento
python train_kpu_depth.py
```

## 📚 Recursos Adicionales

- [Sitio oficial NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [Paper del dataset](http://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf)
- [FastDepth repository](https://github.com/dwofk/fast-depth) (preprocesamiento similar)
- [Documentación de Hugging Face Datasets](https://huggingface.co/docs/datasets/)

---

**Nota**: El dataset completo requiere ~35GB de espacio. Para desarrollo inicial, se puede usar un subset o datos dummy.