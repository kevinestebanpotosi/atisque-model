#!/bin/bash

# Script de configuración para KPU Depth Estimation Project
# Este script ayuda a configurar el entorno y descargar recursos necesarios

set -e  # Salir en caso de error

echo "=============================================="
echo "  Configuración de KPU Depth Estimation Project"
echo "=============================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar si estamos en el directorio correcto
if [ ! -f "pyproject.toml" ]; then
    print_error "No se encontró pyproject.toml. Ejecuta este script desde el directorio del proyecto."
    exit 1
fi

# Crear estructura de directorios
print_info "Creando estructura de directorios..."
mkdir -p data
mkdir -p models
mkdir -p scripts
mkdir -p results
mkdir -p data/calibration

print_success "Estructura de directorios creada"

# Verificar Docker
print_info "Verificando Docker..."
if command -v docker &> /dev/null; then
    print_success "Docker encontrado"
    
    # Verificar Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose encontrado"
    else
        print_warning "Docker Compose no encontrado. Se recomienda instalarlo."
    fi
else
    print_warning "Docker no encontrado. Es necesario para compilar modelos para KPU."
    echo "Instrucciones de instalación:"
    echo "  Windows: https://docs.docker.com/desktop/install/windows-install/"
    echo "  macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "  Linux: https://docs.docker.com/engine/install/"
fi

# Verificar Python
print_info "Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION encontrado"
    
    # Verificar versión mínima
    REQUIRED_VERSION="3.10"
    if python3 -c "import sys; sys.exit(0 if tuple(map(int, sys.version.split()[0].split('.'))) >= tuple(map(int, '$REQUIRED_VERSION'.split('.'))) else 1)"; then
        print_success "Versión de Python compatible ($REQUIRED_VERSION+)"
    else
        print_warning "Se requiere Python $REQUIRED_VERSION o superior"
    fi
else
    print_warning "Python 3 no encontrado"
fi

# Instalación local (opcional)
print_info "¿Deseas instalar las dependencias localmente? (y/n)"
read -r INSTALL_LOCAL

if [[ "$INSTALL_LOCAL" =~ ^[Yy]$ ]]; then
    print_info "Instalando dependencias Python..."
    
    # Verificar si uv está disponible
    if command -v uv &> /dev/null; then
        print_info "Usando uv para instalación..."
        uv pip install -e .
    else
        print_info "Usando pip para instalación..."
        pip install -e .
    fi
    
    print_success "Dependencias instaladas localmente"
fi

# Información sobre el dataset
print_info "Información sobre el dataset NYU Depth V2:"
echo ""
echo "El proyecto requiere el dataset NYU Depth V2 preprocesado."
echo "Archivos necesarios en ./data/:"
echo "  train-000000.tar a train-000011.tar (12 archivos)"
echo "  val-000000.tar a val-000001.tar (2 archivos)"
echo ""
echo "Opciones para obtener el dataset:"
echo "1. Descargar manualmente:"
echo "   https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"
echo ""
echo "2. Usar la versión de Hugging Face (requiere ~35GB):"
echo "   from datasets import load_dataset"
echo "   dataset = load_dataset('sayakpaul/nyu_depth_v2')"
echo ""
print_warning "Nota: Sin el dataset, el script usará datos dummy para demostración."

# Scripts de ayuda
print_info "Creando scripts de ayuda..."

# Script para entrenamiento
cat > scripts/train.sh << 'EOF'
#!/bin/bash
# Script para entrenar el modelo usando Docker

echo "Iniciando entrenamiento con Docker..."
docker-compose up kpu-training
EOF

# Script para compilación
cat > scripts/compile.sh << 'EOF'
#!/bin/bash
# Script para compilar modelo a kmodel

MODEL_INPUT=${1:-"models/kpu_depth_model.onnx"}
MODEL_OUTPUT=${2:-"models/kpu_depth.kmodel"}

echo "Compilando $MODEL_INPUT a $MODEL_OUTPUT..."

docker-compose run --rm nncase-compiler \
  nncase compile "$MODEL_INPUT" \
    --target k210 \
    --input-type float32 \
    --output-type float32 \
    --input-shape [1,3,224,224] \
    --output-kmodel "$MODEL_OUTPUT"
EOF

# Script para ejecutar contenedor interactivo
cat > scripts/shell.sh << 'EOF'
#!/bin/bash
# Script para abrir shell en contenedor

echo "Abriendo shell interactivo en contenedor de compilación..."
docker-compose run --rm nncase-compiler bash
EOF

chmod +x scripts/*.sh

print_success "Scripts de ayuda creados en scripts/"

# Crear README rápido
print_info "Creando archivo de inicio rápido..."

cat > QUICKSTART.md << 'EOF'
# Inicio Rápido - KPU Depth Estimation

## Opción 1: Usar Docker (Recomendado)

```bash
# Construir y ejecutar contenedor de entrenamiento
./scripts/train.sh

# O ejecutar manualmente:
docker-compose up kpu-training
```

## Opción 2: Compilar modelo existente

```bash
# Compilar modelo ONNX a kmodel
./scripts/compile.sh

# Con rutas personalizadas:
./scripts/compile.sh models/mi_modelo.onnx models/mi_modelo.kmodel
```

## Opción 3: Shell interactivo

```bash
# Abrir shell en contenedor con nncase
./scripts/shell.sh
```

## Opción 4: Instalación local

```bash
# Instalar dependencias
pip install -e .

# Ejecutar entrenamiento (requiere dataset)
python train_kpu_depth.py
```

## Estructura del proyecto

- `data/`: Dataset NYU Depth V2 (descargar manualmente)
- `models/`: Modelos ONNX y kmodel generados
- `scripts/`: Scripts de ayuda
- `results/`: Resultados de inferencia
- `docker-compose.yml`: Configuración de servicios Docker
- `Dockerfile`: Imagen Docker con nncase v2.9.0

## Comandos útiles

```bash
# Verificar que Docker funciona
docker --version
docker-compose --version

# Limpiar contenedores Docker
docker-compose down

# Ver logs del entrenamiento
docker-compose logs -f kpu-training

# Ejecutar comando específico en contenedor
docker-compose run --rm nncase-compiler nncase --help
```

## Solución de problemas

1. **Error de dataset**: Asegúrate de tener los archivos .tar en `data/`
2. **Error de memoria**: Reduce batch size en `train_kpu_depth.py`
3. **Error de nncase**: Verifica que el ONNX tenga formas estáticas y ReLU6

## Enlaces importantes

- Documentación completa: `README_NEW.md`
- Guía de entrenamiento: `TRAINING_GUIDE.md`
- Dataset card: `DATASET_CARD.md`
- Licencia: `LICENSE`
EOF

print_success "Archivo QUICKSTART.md creado"

# Resumen final
echo ""
echo "=============================================="
echo "  CONFIGURACIÓN COMPLETADA"
echo "=============================================="
echo ""
echo "📁 Estructura creada:"
echo "   data/          - Dataset NYU Depth V2"
echo "   models/        - Modelos ONNX y kmodel"
echo "   scripts/       - Scripts de ayuda"
echo "   results/       - Resultados"
echo ""
echo "🚀 Para comenzar:"
echo "   1. Revisa QUICKSTART.md para instrucciones rápidas"
echo "   2. Descarga el dataset NYU Depth V2 en data/ (opcional)"
echo "   3. Ejecuta ./scripts/train.sh para entrenar con Docker"
echo ""
echo "📚 Documentación:"
echo "   - README_NEW.md: Documentación principal"
echo "   - TRAINING_GUIDE.md: Guía técnica detallada"
echo "   - DATASET_CARD.md: Información del dataset"
echo ""
echo "🔧 Herramientas disponibles:"
echo "   ./scripts/train.sh    - Entrenar modelo"
echo "   ./scripts/compile.sh  - Compilar a kmodel"
echo "   ./scripts/shell.sh    - Shell interactivo"
echo ""
echo "=============================================="