# Inicio Rápido - KPU Depth Estimation

Guía rápida para comenzar con el proyecto de estimación de profundidad para KPU.

## 🚀 Comenzar en 5 Minutos

### Opción 1: Usar Docker (Recomendado para principiantes)

```bash
# 1. Dar permisos a los scripts
chmod +x setup.sh scripts/*.sh

# 2. Configurar entorno
./setup.sh

# 3. Entrenar modelo (usa datos dummy si no hay dataset)
./scripts/train.sh

# 4. Compilar a kmodel
./scripts/compile.sh
```

### Opción 2: Desarrollo Local (Para expertos)

```bash
# 1. Instalar dependencias
pip install -e .

# 2. Verificar entorno
python main.py check

# 3. Entrenar modelo
python train_kpu_depth.py

# 4. Usar Docker solo para compilación
docker-compose run --rm nncase-compiler \
  nncase compile kpu_depth_model.onnx \
    --target k210 \
    --input-type float32 \
    --output-type float32 \
    --input-shape [1,3,224,224] \
    --output-kmodel kpu_depth.kmodel
```

## 📋 Requisitos Mínimos

- **Docker** o **Docker Desktop** (recomendado)
- **Python 3.10+** (opcional, para desarrollo local)
- **8GB RAM** (16GB recomendado para entrenamiento)
- **10GB espacio libre** en disco

## 📁 Estructura Básica

```
# Directorios importantes
data/           # Dataset NYU Depth V2 (descargar manualmente)
models/         # Modelos generados (.onnx, .kmodel)
scripts/        # Scripts de ayuda
results/        # Resultados de inferencia

# Archivos principales
Readme.md       # Documentación completa
QUICKSTART.md   # Esta guía
TRAINING_GUIDE.md # Guía técnica de entrenamiento
COMPILATION_GUIDE.md # Guía de compilación
DATA_PREPARATION.md # Preparación de datos
```

## 🔧 Comandos Esenciales

### Verificación
```bash
# Verificar que todo funciona
./setup.sh

# O verificar manualmente
python main.py check
```

### Entrenamiento
```bash
# Con Docker (recomendado)
./scripts/train.sh

# Manual con Docker
docker-compose up kpu-training

# Local (requiere dataset)
python train_kpu_depth.py
```

### Compilación
```bash
# Compilar modelo existente
./scripts/compile.sh

# Con rutas personalizadas
./scripts/compile.sh mi_modelo.onnx mi_modelo.kmodel
```

### Desarrollo
```bash
# Shell interactiva con nncase
./scripts/shell.sh

# Limpiar contenedores
docker-compose down
```

## 📊 Dataset NYU Depth V2

### Opción Rápida (Datos Dummy)
El proyecto funciona sin dataset usando datos dummy para demostración.

### Opción Completa (Dataset Real)
1. Descargar de: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Extraer archivos .tar a `data/`
3. Se requieren 12 archivos train-*.tar y 2 archivos val-*.tar

### Estructura esperada:
```
data/
├── train-000000.tar
├── train-000001.tar
├── ...
├── train-000011.tar
├── val-000000.tar
└── val-000001.tar
```

## 🎯 Flujo de Trabajo Típico

### 1. Desarrollo Inicial
```bash
# Usar datos dummy para prototipado
./scripts/train.sh   # Entrena con datos dummy
./scripts/compile.sh # Compila a kmodel
```

### 2. Entrenamiento Real
1. Descargar dataset NYU Depth V2
2. Colocar en `data/`
3. Ejecutar `./scripts/train.sh`
4. Ajustar hiperparámetros si es necesario

### 3. Optimización
1. Crear dataset de calibración en `data/calibration/`
2. Compilar con cuantización: `./scripts/compile.sh`
3. Validar precisión vs velocidad

### 4. Despliegue
1. Probar kmodel en hardware K210
2. Ajustar arquitectura si es necesario
3. Documentar resultados

## 🐛 Solución de Problemas Comunes

### "docker-compose no encontrado"
```bash
# Instalar Docker Desktop o docker-compose
# Windows/macOS: https://docs.docker.com/desktop/
# Linux: sudo apt install docker-compose
```

### "No se encontró kpu_depth_model.onnx"
```bash
# Ejecutar entrenamiento primero
./scripts/train.sh
```

### "Dataset no encontrado"
```bash
# Usar datos dummy (el script lo hace automáticamente)
# O descargar dataset real
```

### "Error de memoria"
```bash
# Reducir batch size en train_kpu_depth.py
# Usar GPU si está disponible
# Aumentar swap/memoria virtual
```

## 📈 Verificación de Resultados

### Modelo ONNX Generado
```bash
# Verificar que se creó
ls -lh kpu_depth_model.onnx

# Tamaño típico: 10-50MB
```

### KModel Generado
```bash
# Verificar compilación
ls -lh *.kmodel

# Probar inferencia (requiere imágenes de prueba)
docker-compose run --rm nncase-compiler \
  nncase infer kpu_depth.kmodel \
    --dataset test_images/ \
    --output results/
```

## 🔗 Enlaces Rápidos

- **Documentación completa**: `Readme.md`
- **Guía técnica**: `TRAINING_GUIDE.md`
- **Compilación**: `COMPILATION_GUIDE.md`
- **Dataset**: `DATA_PREPARATION.md`
- **Scripts de ayuda**: `scripts/`

## 🆘 Soporte

### Problemas con nncase
1. Verificar que el modelo usa ReLU6, no ReLU
2. Asegurar formas estáticas (no dinámicas)
3. Verificar opset 11/12 en exportación ONNX

### Problemas con Docker
1. Reiniciar Docker Desktop
2. Verificar que los volúmenes están montados
3. Limpiar cache: `docker system prune`

### Problemas de rendimiento
1. Reducir resolución del modelo
2. Usar más convoluciones separables
3. Aplicar cuantización

---

**💡 Consejo**: Comienza con datos dummy para familiarizarte, luego descarga el dataset real para entrenamiento completo.

**⏱️ Tiempo estimado**: 
- Configuración inicial: 5-10 minutos
- Entrenamiento con datos dummy: 2-5 minutos  
- Entrenamiento con dataset real: 30+ minutos
- Compilación: 1-2 minutos