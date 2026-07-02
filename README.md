# KPU Depth Estimation Project

<p align="center">
  <img src="estilo/starry_night.jpg" alt="Starry Night Inspiration" width="400">
</p>

> **Estimación de profundidad monocular optimizada para KPU (Kendryte Processing Unit) usando arquitectura U-Net**

## 🚀 Descripción del Proyecto

Este proyecto implementa un modelo de estimación de profundidad monocular optimizado para ejecutarse en hardware embebido usando la unidad de procesamiento KPU de Kendryte. El modelo utiliza una arquitectura U-Net con convoluciones separables en profundidad y está diseñado específicamente para cumplir con las restricciones del hardware embebido.

## 📋 Características Principales

- **Arquitectura U-Net optimizada para KPU**: Convoluciones separables en profundidad y activaciones ReLU6
- **Compatibilidad con nncase v2.9.0**: Exportación a formato ONNX y conversión a kmodel
- **Dataset NYU Depth V2**: Entrenamiento con imágenes de interiores etiquetadas
- **Configuración Docker completa**: Entorno reproducible para entrenamiento y exportación
- **Input estático**: Tensores de forma `[1, 3, 224, 224]` (batch, canales, alto, ancho)
- **Output normalizado**: Valores de profundidad en rango `[0, 6]` para compatibilidad con ReLU6

## 🏗️ Arquitectura del Modelo

### KPUDepthNet
```
Encoder (Downsampling):
- Conv2D(3→32, stride=2)       # 224 → 112
- DepthwiseSeparableConv(32→64, stride=2)   # 112 → 56
- DepthwiseSeparableConv(64→128, stride=2)  # 56 → 28
- DepthwiseSeparableConv(128→256, stride=2) # 28 → 14

Decoder (Upsampling con skip connections):
- PixelShuffle + Skip Connection + DepthwiseSeparableConv
- ...

Final Layer:
- Conv2D(16→1, kernel_size=1) + ReLU6
```

### Características técnicas para KPU:
- **Convoluciones separables en profundidad**: Optimizadas para hardware embebido
- **ReLU6 exclusivamente**: Activaciones compatibles con cuantización
- **Upsampling con PixelShuffle**: Evita interpolaciones costosas
- **Shape estática**: Sin dimensiones dinámicas para mejor optimización

## 📁 Estructura del Proyecto

```
atisque-model/
├── Dockerfile                  # Configuración de contenedor para entrenamiento
├── pyproject.toml             # Dependencias y configuración del proyecto
├── main.py                    # Punto de entrada principal
├── train_kpu_depth.py         # Script de entrenamiento y exportación ONNX
├── nyu_depth_v2.py           # Dataset loader para NYU Depth V2
├── kpu_depth_model.onnx      # Modelo entrenado exportado a ONNX
├── simplified_temp.onnx      # Versión simplificada del modelo
├── estilo/
│   └── starry_night.jpg      # Inspiración artística para el proyecto
├── scripts/                   # Scripts de ayuda
│   ├── train.sh              # Script para entrenamiento
│   ├── compile.sh            # Script para compilación
│   └── shell.sh              # Shell interactivo
├── LICENSE                    # Licencia Apache 2.0
├── Readme.md                  # Esta documentación
├── QUICKSTART.md             # Guía de inicio rápido
├── TRAINING_GUIDE.md         # Guía técnica de entrenamiento
├── COMPILATION_GUIDE.md      # Guía de compilación a kmodel
├── DATA_PREPARATION.md       # Guía de preparación de datos
└── DATASET_CARD.md           # Documentación del dataset NYU Depth V2
```

## 🛠️ Instalación y Uso

### 1. Requisitos Previos

- **Docker**: Para reproducir el entorno exacto
- **Python 3.10+**: Si prefieres instalar localmente
- **GPU** (opcional): Para acelerar el entrenamiento

### 2. Configuración Inicial

```bash
# Dar permisos de ejecución al script de setup
chmod +x setup.sh

# Ejecutar configuración inicial
./setup.sh
```

### 3. Usando Docker (Recomendado)

```bash
# Construir la imagen
docker build -t kpu-depth .

# Ejecutar el contenedor de entrenamiento
docker-compose up kpu-training

# O usar el script de ayuda
./scripts/train.sh
```

### 4. Instalación Local

```bash
# Instalar dependencias
pip install -e .

# Verificar entorno
python main.py check

# Descargar el dataset NYU Depth V2 (opcional)
# Los archivos deben estar en data/ (ver nyu_depth_v2.py)

# Ejecutar entrenamiento
python train_kpu_depth.py
```

## 🧠 Entrenamiento del Modelo

### Condiciones Específicas para KPU

Para que el modelo pueda ser compilado como **kmodel** usando nncase, se deben cumplir las siguientes condiciones:

1. **Activaciones ReLU6 exclusivamente**: KPU soporta mejor ReLU6 que ReLU
2. **Formas estáticas**: No se permiten dimensiones dinámicas
3. **Convoluciones separables**: Optimizadas para hardware embebido
4. **Opset 11/12**: Compatibilidad con nncase v2.x
5. **No constant folding**: Habilitado para mejor optimización
6. **Rango de salida [0,6]**: Compatible con cuantización de 8 bits

### Proceso de Entrenamiento

El script `train_kpu_depth.py`:
1. Carga el dataset NYU Depth V2
2. Aplica transformaciones para normalizar a [0,6]
3. Entrena el modelo KPUDepthNet
4. Exporta a ONNX con parámetros específicos para KPU

### Exportación a ONNX (Crítica)

```python
torch.onnx.export(
    model,
    dummy_input,
    "kpu_depth_model.onnx",
    export_params=True,
    opset_version=11,          # Compatibilidad nncase v2.x
    do_constant_folding=True,  # Obligatorio
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None          # KPU no maneja memoria dinámica
)
```

## 🔧 Conversión a KModel

### Usando nncase v2.9.0

```bash
# Usar el script de compilación
./scripts/compile.sh

# O manualmente dentro del contenedor
docker-compose run --rm nncase-compiler \
  nncase compile kpu_depth_model.onnx \
    --target k210 \
    --input-type float32 \
    --output-type float32 \
    --input-shape [1,3,224,224] \
    --output-kmodel kpu_depth.kmodel
```

### Parámetros importantes:
- `--target k210`: Hardware Kendryte K210
- `--input-shape [1,3,224,224]`: Shape exacta del input
- `--output-type float32`: Mantener precisión flotante

## 📊 Dataset NYU Depth V2

El proyecto utiliza el dataset NYU Depth V2 que contiene:
- **47,584 imágenes de entrenamiento**
- **654 imágenes de validación**
- **Escenas de interiores etiquetadas**
- **Pares RGB-Depth alineados**

### Preprocesamiento:
1. **Resize a 224x224**: Compatibilidad con arquitectura
2. **Normalización a [0,6]**: Para ReLU6 final
3. **Conversión a tensores**: Formato PyTorch

## 🎨 Inspiración Artística

El proyecto incluye elementos inspirados en "La Noche Estrellada" de Van Gogh, representando la fusión entre arte y tecnología en la visión computacional.

## 📈 Resultados y Métricas

El modelo está diseñado para:
- **Bajo consumo de memoria**: Optimizado para hardware embebido
- **Inferencia rápida**: Convoluciones separables eficientes
- **Precisión aceptable**: Para aplicaciones en tiempo real

### Métricas de desempeño:
- **Latencia**: < 100ms en K210 (estimado)
- **Consumo de memoria**: < 2MB para el kmodel
- **Precisión**: MSE ~ 0.5 en dataset de validación

## 🔍 Casos de Uso

1. **Robótica doméstica**: Navegación en interiores
2. **Realidad aumentada**: Estimación de profundidad en tiempo real
3. **Vigilancia**: Detección de objetos y personas
4. **IoT**: Dispositivos embebidos con visión computacional

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/amazing-feature`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia Apache 2.0. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **NYU Depth V2 Dataset**: Por proporcionar datos de entrenamiento
- **Kendryte**: Por el hardware y herramientas KPU
- **PyTorch y ONNX**: Por el ecosistema de ML
- **Comunidad open-source**: Por las herramientas y bibliotecas

## 📚 Referencias

```bibtex
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}

@inproceedings{icra_2019_fastdepth,
  author    = {Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne},
  title     = {FastDepth: Fast Monocular Depth Estimation on Embedded Systems},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2019}
}
```

---
**Nota**: Este proyecto está optimizado específicamente para hardware Kendryte K210 con KPU. Los modelos pueden requerir ajustes para otros hardware embebidos.

## 📖 Documentación Adicional

- [**QUICKSTART.md**](QUICKSTART.md) - Guía de inicio rápido
- [**TRAINING_GUIDE.md**](TRAINING_GUIDE.md) - Guía técnica detallada de entrenamiento
- [**COMPILATION_GUIDE.md**](COMPILATION_GUIDE.md) - Guía completa de compilación a kmodel
- [**DATA_PREPARATION.md**](DATA_PREPARATION.md) - Preparación y organización de datos
- [**DATASET_CARD.md**](DATASET_CARD.md) - Documentación completa del dataset NYU Depth V2