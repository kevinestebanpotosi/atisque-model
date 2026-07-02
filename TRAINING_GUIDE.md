# Guía de Entrenamiento y Compilación para KPU

Esta guía detalla el proceso completo para entrenar un modelo de estimación de profundidad y compilarlo para KPU (Kendryte Processing Unit).

## 📋 Requisitos Técnicos para KPU

### Restricciones de Hardware
1. **Formas estáticas**: El modelo debe tener dimensiones fijas
2. **ReLU6 exclusivamente**: Las activaciones deben usar ReLU6, no ReLU
3. **Convoluciones optimizadas**: Preferir convoluciones separables en profundidad
4. **Upsampling específico**: Usar PixelShuffle en lugar de interpolaciones
5. **Rango de salida limitado**: Valores entre 0 y 6 para compatibilidad con cuantización

### Parámetros de ONNX para nncase
```python
# Parámetros CRÍTICOS para exportación
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,          # Compatible con nncase v2.x
    do_constant_folding=True,  # OBLIGATORIO para optimización
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None          # SIN dimensiones dinámicas
)
```

## 🧠 Arquitectura del Modelo KPUDepthNet

### Componentes Principales

#### 1. Convolución Separable en Profundidad (DepthwiseSeparableConv)
```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise convolution (1 grupo por canal)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6(inplace=True)
```

#### 2. Encoder (Downsampling)
- **Conv2D inicial**: 3→32 canales, stride=2 (224→112)
- **Tres bloques separables**: Con reducción espacial 2x cada uno
- **Total**: 4 niveles de reducción (224→14)

#### 3. Decoder con Skip Connections (Upsampling)
- **PixelShuffle**: Factor 2 de upsampling
- **Concatenación**: Conexiones U-Net desde el encoder
- **Bloques separables**: Después de cada concatenación

#### 4. Capa Final
- **Conv 1x1**: 16→1 canal
- **ReLU6**: Limita el output a [0,6]

## 📊 Preparación del Dataset

### Dataset NYU Depth V2
El dataset debe descargarse y organizarse en la estructura:

```
data/
├── train-000000.tar
├── train-000001.tar
├── ...
├── val-000000.tar
└── val-000001.tar
```

### Transformaciones Aplicadas
1. **Resize a 224x224**: Formato compatible con la arquitectura
2. **Conversión a tensor**: Para PyTorch
3. **Normalización de profundidad**: Ajuste a rango [0,6]
```python
# Normalización crítica para KPU
d_max = depth_t.max()
if d_max > 0:
    depth_t = (depth_t / d_max) * 6.0
```

## 🚀 Proceso de Entrenamiento

### 1. Inicialización
```python
device = torch.device('cuda' if torch.cuda.is_available() 
                     else 'mps' if torch.backends.mps.is_available() 
                     else 'cpu')
model = KPUDepthNet().to(device)
```

### 2. Hiperparámetros Recomendados
- **Batch size**: 4-8 (dependiendo de la memoria)
- **Learning rate**: 1e-3 a 1e-4
- **Optimizador**: Adam o SGD con momentum
- **Función de pérdida**: L1Loss o SmoothL1Loss

### 3. Loop de Entrenamiento
```python
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        images = batch['pixel_values'].to(device)
        depths = batch['labels'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, depths)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🔧 Exportación a ONNX

### Pasos Críticos
1. **Modelo en modo evaluación**: `model.eval()`
2. **CPU para exportación**: `model.cpu()`
3. **Input dummy**: Exactamente `[1, 3, 224, 224]`
4. **Parámetros específicos**: Ver sección anterior

### Verificación del ONNX
```bash
# Verificar que el modelo se exportó correctamente
python -c "import onnx; model = onnx.load('kpu_depth_model.onnx'); print('Input:', model.graph.input[0])"
```

## 🛠️ Compilación a KModel con nncase

### Instalación de nncase v2.9.0
```bash
# Usando pip/uv (en el contenedor Docker)
pip install uv
uv pip install --system \
    https://github.com/kendryte/nncase/releases/download/v2.9.0/nncase-2.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    https://github.com/kendryte/nncase/releases/download/v2.9.0/nncase_kpu-2.9.0-py2.py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### Comando de Compilación
```bash
nncase compile kpu_depth_model.onnx \
  --target k210 \
  --input-type float32 \
  --output-type float32 \
  --input-shape [1,3,224,224] \
  --output-kmodel kpu_depth.kmodel \
  --dataset data/calibration \
  --calibrate-method no_clip
```

### Parámetros Explicados
- `--target k210`: Hardware Kendryte K210
- `--input-shape [1,3,224,224]`: Forma exacta del input
- `--output-type float32`: Mantener precisión (puede ser uint8 para cuantización)
- `--dataset`: Dataset para calibración de cuantización
- `--calibrate-method`: Método de calibración

## 🔍 Validación del KModel

### 1. Inferencia de Prueba
```python
import nncase
import numpy as np

# Cargar el kmodel
with open('kpu_depth.kmodel', 'rb') as f:
    kmodel = f.read()

# Crear intérprete
interpreter = nncase.Interpreter()
interpreter.load_model(kmodel)

# Preparar input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Ejecutar inferencia
interpreter.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_data))
interpreter.run()
output = interpreter.get_output_tensor(0).to_numpy()

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
```

### 2. Verificaciones de Calidad
- **Rango de salida**: Debe estar entre 0 y 6
- **Forma correcta**: `[1, 1, 224, 224]`
- **Sin NaN/Inf**: Valores finitos
- **Consistencia**: Comparar con salida ONNX

## 🐛 Solución de Problemas Comunes

### Error: "Dynamic shapes not supported"
**Causa**: El modelo tiene dimensiones dinámicas
**Solución**: Asegurar que `dynamic_axes=None` en exportación ONNX

### Error: "Unsupported operation"
**Causa**: Operación no soportada por KPU
**Solución**: 
- Reemplazar ReLU con ReLU6
- Usar convoluciones separables
- Evitar operaciones complejas (interpolaciones, padding asimétrico)

### Error: "Quantization failed"
**Causa**: Rango de valores fuera de lo esperado
**Solución**: 
- Asegurar que las activaciones usen ReLU6
- Verificar que la salida esté en [0,6]
- Usar dataset de calibración representativo

### Rendimiento Bajo
**Optimizaciones**:
- Reducir número de parámetros
- Usar más convoluciones separables
- Aumentar stride en capas tempranas
- Reducir precisión a int8 (cuantización)

## 📈 Métricas de Desempeño

### Objetivos para KPU K210
- **Memoria del modelo**: < 2MB
- **Latencia de inferencia**: < 100ms
- **Consumo de energía**: < 300mW
- **Precisión**: MSE < 1.0 en NYU Depth V2

### Herramientas de Perfilado
```bash
# Perfilar el kmodel
nncase infer kpu_depth.kmodel --dataset test_images/ --output results/

# Analizar rendimiento
nncase eval kpu_depth.kmodel --target k210 --benchmark
```

## 🔄 Flujo de Trabajo Recomendado

1. **Desarrollo local**: Entrenar y probar en GPU/CPU
2. **Exportación ONNX**: Verificar compatibilidad con nncase
3. **Compilación Docker**: Usar el contenedor para compilación
4. **Pruebas en hardware**: Deploy en K210 real
5. **Iteración**: Ajustar arquitectura según resultados

## 📚 Recursos Adicionales

- [Documentación nncase](https://github.com/kendryte/nncase)
- [Kendryte K210 Datasheet](https://canaan.io/product/kendryteai)
- [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [ONNX Operators Supported by nncase](https://github.com/kendryte/nncase/blob/master/docs/operators.md)

---

**Nota**: Esta guía está optimizada para nncase v2.9.0 y hardware K210. Versiones futuras pueden requerir ajustes.