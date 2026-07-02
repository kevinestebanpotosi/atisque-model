#!/bin/bash

# Script para compilar modelo ONNX a kmodel

echo "=========================================="
echo "  Compilación de modelo a KModel"
echo "=========================================="
echo ""

# Parámetros
MODEL_INPUT=${1:-"kpu_depth_model.onnx"}
MODEL_OUTPUT=${2:-"kpu_depth.kmodel"}

# Verificar archivo de entrada
echo "📄 Verificando archivo de entrada..."
if [ ! -f "$MODEL_INPUT" ]; then
    echo "❌ Error: No se encontró $MODEL_INPUT"
    echo ""
    echo "📝 Soluciones:"
    echo "   1. Ejecuta primero el entrenamiento: ./scripts/train.sh"
    echo "   2. Usa un modelo existente: ./scripts/compile.sh mi_modelo.onnx"
    echo "   3. Verifica la ruta del archivo"
    exit 1
fi

size_mb=$(du -k "$MODEL_INPUT" | cut -f1)
size_mb=$((size_mb / 1024))
echo "✅ Modelo encontrado: $MODEL_INPUT (${size_mb} MB)"
echo "📤 Salida: $MODEL_OUTPUT"

# Verificar docker-compose
echo ""
echo "🐳 Verificando Docker..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Error: docker-compose no encontrado"
    exit 1
fi

# Verificar dataset de calibración (opcional)
echo ""
echo "📊 Dataset de calibración..."
if [ -d "data/calibration" ] && [ -n "$(ls -A data/calibration 2>/dev/null)" ]; then
    echo "✅ Dataset de calibración encontrado"
    CALIBRATION_OPTIONS="--dataset data/calibration --calibrate-method no_clip"
    OUTPUT_TYPE="uint8"
    echo "   Se usará cuantización uint8"
else
    echo "⚠️  Dataset de calibración no encontrado"
    CALIBRATION_OPTIONS=""
    OUTPUT_TYPE="float32"
    echo "   Se usará precisión float32"
    echo "   Para cuantización, crea data/calibration/ con imágenes de calibración"
fi

# Comando de compilación
echo ""
echo "🔧 Compilando modelo..."
echo "   Target: k210"
echo "   Input shape: [1,3,224,224]"
echo "   Output type: $OUTPUT_TYPE"
echo ""

COMPILE_CMD="nncase compile \"$MODEL_INPUT\" \
  --target k210 \
  --input-type float32 \
  --output-type $OUTPUT_TYPE \
  --input-shape [1,3,224,224] \
  $CALIBRATION_OPTIONS \
  --output-kmodel \"$MODEL_OUTPUT\""

echo "Comando ejecutado:"
echo "  $COMPILE_CMD"
echo ""

# Ejecutar compilación
if docker-compose version &> /dev/null; then
    docker-compose run --rm nncase-compiler \
      bash -c "$COMPILE_CMD"
elif docker compose version &> /dev/null; then
    docker compose run --rm nncase-compiler \
      bash -c "$COMPILE_CMD"
else
    echo "❌ No se pudo ejecutar docker-compose"
    exit 1
fi

# Verificar resultado
echo ""
echo "=========================================="
if [ -f "$MODEL_OUTPUT" ]; then
    size_mb=$(du -k "$MODEL_OUTPUT" | cut -f1)
    size_mb=$((size_mb / 1024))
    echo "✅ Compilación exitosa"
    echo "📁 KModel generado: $MODEL_OUTPUT"
    echo "📏 Tamaño: ${size_mb} MB"
    echo "🎯 Tipo: $OUTPUT_TYPE"
    echo ""
    echo "🔍 Para probar el modelo:"
    echo "   ./scripts/shell.sh"
    echo "   # Dentro del contenedor:"
    echo "   nncase infer $MODEL_OUTPUT --dataset test_images/ --output results/"
else
    echo "❌ Error en la compilación"
    echo "   Verifica los logs anteriores"
    echo "   Posibles problemas:"
    echo "   - Operaciones no soportadas por KPU"
    echo "   - Formas dinámicas en el modelo"
    echo "   - Falta de memoria"
fi
echo "=========================================="