#!/bin/bash

# Script para abrir shell interactivo en contenedor

echo "=========================================="
echo "  Shell Interactivo - KPU Development"
echo "=========================================="
echo ""

# Verificar docker-compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Error: docker-compose no encontrado"
    exit 1
fi

echo "🐳 Iniciando contenedor interactivo..."
echo ""
echo "📚 Herramientas disponibles:"
echo "  nncase --help          - Ver ayuda de nncase"
echo "  python --version       - Ver Python"
echo "  python train_kpu_depth.py - Entrenar modelo"
echo ""
echo "📁 Directorio actual montado en /app"
echo ""

# Información útil
echo "🔧 Comandos útiles de nncase:"
echo "   # Compilar modelo"
echo "   nncase compile kpu_depth_model.onnx \\"
echo "     --target k210 \\"
echo "     --input-type float32 \\"
echo "     --output-type float32 \\"
echo "     --input-shape [1,3,224,224] \\"
echo "     --output-kmodel kpu_depth.kmodel"
echo ""
echo "   # Inferencia con kmodel"
echo "   nncase infer kpu_depth.kmodel \\"
echo "     --dataset data/calibration/ \\"
echo "     --output results/"
echo ""
echo "   # Benchmark"
echo "   nncase eval kpu_depth.kmodel \\"
echo "     --target k210 \\"
echo "     --benchmark"
echo ""

# Ejecutar shell interactiva
echo "🚀 Iniciando shell bash..."
echo "=========================================="
echo ""

if docker-compose version &> /dev/null; then
    docker-compose run --rm nncase-compiler bash
elif docker compose version &> /dev/null; then
    docker compose run --rm nncase-compiler bash
else
    echo "❌ No se pudo ejecutar docker-compose"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Shell finalizado"
echo "=========================================="