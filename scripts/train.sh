#!/bin/bash

# Script para entrenar el modelo usando Docker

echo "=========================================="
echo "  Iniciando entrenamiento KPU Depth Model"
echo "=========================================="
echo ""

# Verificar que docker-compose está disponible
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Error: docker-compose no encontrado"
    echo "Instala docker-compose o usa Docker Desktop"
    exit 1
fi

# Verificar estructura de datos
echo "📊 Verificando estructura de datos..."
if [ -d "data" ] && [ -n "$(ls -A data 2>/dev/null)" ]; then
    echo "✅ Dataset encontrado en data/"
    echo "   Archivos: $(ls data/*.tar 2>/dev/null | wc -l | xargs) .tar"
else
    echo "⚠️  Advertencia: Dataset no encontrado o vacío"
    echo "   El script usará datos dummy para demostración"
    echo "   Para entrenamiento real, descarga el dataset NYU Depth V2:"
    echo "   https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"
    echo ""
    read -p "¿Continuar con datos dummy? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Entrenamiento cancelado"
        exit 0
    fi
fi

# Iniciar entrenamiento
echo ""
echo "🚀 Iniciando entrenamiento con Docker..."
echo "   Esto puede tomar varios minutos..."
echo ""

# Usar docker-compose
if docker-compose version &> /dev/null; then
    docker-compose up kpu-training
elif docker compose version &> /dev/null; then
    docker compose up kpu-training
else
    echo "❌ No se pudo ejecutar docker-compose"
    exit 1
fi

# Verificar resultado
echo ""
echo "=========================================="
if [ -f "kpu_depth_model.onnx" ]; then
    size_mb=$(du -k "kpu_depth_model.onnx" | cut -f1)
    size_mb=$((size_mb / 1024))
    echo "✅ Entrenamiento completado exitosamente"
    echo "📁 Modelo generado: kpu_depth_model.onnx"
    echo "📏 Tamaño: ${size_mb} MB"
    echo ""
    echo "🔧 Para compilar a kmodel:"
    echo "   ./scripts/compile.sh"
else
    echo "⚠️  Entrenamiento completado, pero no se generó el modelo"
    echo "   Verifica los logs anteriores para errores"
fi
echo "=========================================="