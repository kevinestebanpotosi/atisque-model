# Guía de Compilación a KModel

Esta guía detalla el proceso de compilar modelos ONNX a formato kmodel para ejecución en hardware Kendryte K210 con KPU.

## 📋 Prerrequisitos

### 1. Entorno Docker
El proyecto incluye un Dockerfile con todas las dependencias necesarias:

```bash
# Construir la imagen
docker build -t kpu-nncase .

# O usar docker-compose
docker-compose build
```

### 2. Modelo ONNX Preparado
El modelo debe cumplir con:
- **Formas estáticas**: Sin dimensiones dinámicas
- **ReLU6 exclusivamente**: No usar ReLU normal
- **Compatibilidad con opset 11/12**: Para nncase v2.9.0
- **Input shape**: `[1, 3, 224, 224]` (batch, channels, height, width)

## 🛠️ Compilación Básica

### Comando Principal
```bash
nncase compile kpu_depth_model.onnx \
  --target k210 \
  --input-type float32 \
  --output-type float32 \
  --input-shape [1,3,224,224] \
  --output-kmodel kpu_depth.kmodel
```

### Parámetros Explicados

| Parámetro | Descripción | Valor Recomendado |
|-----------|-------------|-------------------|
| `--target` | Hardware objetivo | `k210` |
| `--input-type` | Tipo de datos de entrada | `float32` o `uint8` |
| `--output-type` | Tipo de datos de salida | `float32` o `uint8` |
| `--input-shape` | Forma exacta del input | `[1,3,224,224]` |
| `--output-kmodel` | Ruta de salida | `.kmodel` |

## 🔧 Compilación Avanzada

### 1. Compilación con Cuantización
Para reducir el tamaño del modelo y mejorar el rendimiento:

```bash
nncase compile kpu_depth_model.onnx \
  --target k210 \
  --input-type uint8 \
  --output-type uint8 \
  --input-shape [1,3,224,224] \
  --dataset data/calibration \
  --calibrate-method no_clip \
  --output-kmodel kpu_depth_quantized.kmodel
```

**Requisitos para cuantización:**
- Dataset de calibración en `data/calibration/`
- Imágenes representativas del dominio
- Preferiblemente 100-1000 imágenes

### 2. Compilación con Optimizaciones
```bash
nncase compile kpu_depth_model.onnx \
  --target k210 \
  --input-type float32 \
  --output-type float32 \
  --input-shape [1,3,224,224] \
  --dump-ir \
  --dump-asm \
  --dump-import-op-range \
  --output-kmodel kpu_depth_optimized.kmodel
```

**Parámetros de depuración:**
- `--dump-ir`: Volcar representación intermedia
- `--dump-asm`: Volcar código assembly
- `--dump-import-op-range`: Volcar rangos de operadores

## 📊 Validación del KModel

### 1. Inferencia de Prueba
```bash
# Ejecutar inferencia con imágenes de prueba
nncase infer kpu_depth.kmodel \
  --dataset data/test_images \
  --output results/
```

### 2. Benchmark de Rendimiento
```bash
# Medir rendimiento
nncase eval kpu_depth.kmodel \
  --target k210 \
  --benchmark
```

### 3. Validación con Python
```python
import nncase
import numpy as np

# Cargar kmodel
with open('kpu_depth.kmodel', 'rb') as f:
    kmodel = f.read()

# Configurar intérprete
interpreter = nncase.Interpreter()
interpreter.load_model(kmodel)

# Crear input de prueba
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Ejecutar
interpreter.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_data))
interpreter.run()
output = interpreter.get_output_tensor(0).to_numpy()

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
```

## 🐛 Solución de Problemas Comunes

### Error: "Unsupported operation: Resize"
**Causa**: KPU no soporta operaciones de resize/interpolación
**Solución**: 
- Usar PixelShuffle en lugar de interpolaciones
- Rediseñar la arquitectura para evitar resize

### Error: "Shape mismatch"
**Causa**: Formas incompatibles entre operaciones
**Solución**:
- Verificar que todas las formas sean estáticas
- Asegurar que `dynamic_axes=None` en exportación ONNX
- Revisar skip connections en U-Net

### Error: "Quantization range too large"
**Causa**: Valores fuera de rango para cuantización uint8
**Solución**:
- Usar ReLU6 para limitar activaciones
- Aplicar normalización adecuada
- Revisar dataset de calibración

### Error: "Memory allocation failed"
**Causa**: Modelo demasiado grande para KPU
**Solución**:
- Reducir número de canales
- Usar más convoluciones separables
- Aplicar pruning o cuantización

## 📈 Optimización de Rendimiento

### 1. Reducción de Memoria
| Técnica | Reducción Esperada | Impacto en Precisión |
|---------|-------------------|---------------------|
| Cuantización a uint8 | 4x | Bajo-Medio |
| Pruning (50%) | 2x | Medio |
| Convoluciones separables | 1.5-2x | Muy Bajo |

### 2. Mejora de Velocidad
```bash
# Compilar con optimizaciones específicas
nncase compile model.onnx \
  --target k210 \
  --input-type uint8 \
  --output-type uint8 \
  --input-shape [1,3,224,224] \
  --quant-scheme symmetric \
  --w-quant-type asymmetric \
  --dump-asm \
  --output-kmodel model_fast.kmodel
```

### 3. Uso Eficiente de KPU
- **Operaciones soportadas**: Conv2D, DepthwiseConv2D, ReLU6, Add, Concat
- **Operaciones NO soportadas**: Resize, Pad, Transpose complejas
- **Limitaciones**: Memoria limitada (~6MB), Sin soporte para FP16

## 🔄 Flujo de Trabajo Recomendado

### Fase 1: Desarrollo y Prueba
1. Entrenar modelo en GPU/CPU
2. Exportar a ONNX con formas estáticas
3. Probar inferencia ONNX en PC

### Fase 2: Compilación Inicial
1. Compilar a float32 sin optimizaciones
2. Validar funcionalidad básica
3. Verificar formas y rangos

### Fase 3: Optimización
1. Preparar dataset de calibración
2. Compilar versión cuantizada
3. Validar precisión vs float32
4. Ajustar hiperparámetros si es necesario

### Fase 4: Despliegue
1. Benchmark final
2. Documentar métricas de rendimiento
3. Crear script de inferencia
4. Preparar para integración en aplicación

## 📝 Checklist de Compilación

- [ ] Modelo ONNX exportado con `dynamic_axes=None`
- [ ] Todas las activaciones son ReLU6
- [ ] Forma de input: `[1, 3, 224, 224]`
- [ ] No hay operaciones no soportadas
- [ ] Dataset de calibración disponible (para cuantización)
- [ ] Memoria estimada < 6MB
- [ ] Output en rango [0, 6] (para nuestro caso)

## 🔗 Recursos Adicionales

- [Documentación oficial de nncase](https://github.com/kendryte/nncase)
- [Lista de operadores soportados](https://github.com/kendryte/nncase/blob/master/docs/operators.md)
- [Ejemplos de compilación](https://github.com/kendryte/nncase/tree/master/examples)
- [Foro de la comunidad Kendryte](https://forum.kendryte.com/)

---

**Nota**: Esta guía está escrita para nncase v2.9.0. Versiones futuras pueden tener diferencias en los parámetros o capacidades.