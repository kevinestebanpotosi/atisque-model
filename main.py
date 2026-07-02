#!/usr/bin/env python3
"""
Main entry point for KPU Depth Estimation Project.
"""

import argparse
import sys
from pathlib import Path


def check_environment():
    """Check if required dependencies are installed."""
    print("🔍 Verificando entorno...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'onnx': 'ONNX',
        'datasets': 'Hugging Face Datasets',
        'numpy': 'NumPy',
        'h5py': 'HDF5',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name} encontrado")
        except ImportError:
            print(f"  ❌ {name} no encontrado")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Paquetes faltantes: {', '.join(missing)}")
        print("Instalar con: pip install " + " ".join(missing))
        return False
    
    print("\n✅ Entorno verificado correctamente")
    return True


def check_dataset():
    """Check if dataset is available."""
    print("\n📊 Verificando dataset...")
    
    data_dir = Path("data")
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"  ✅ Dataset encontrado en {data_dir}")
        # Count tar files
        tar_files = list(data_dir.glob("*.tar"))
        if tar_files:
            print(f"  📁 {len(tar_files)} archivos .tar encontrados")
        return True
    else:
        print("  ⚠️  Dataset no encontrado en data/")
        print("  ℹ️  Descargar de: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
        return False


def check_models():
    """Check if model files exist."""
    print("\n🤖 Verificando modelos...")
    
    models = {
        'ONNX model': Path("kpu_depth_model.onnx"),
        'Simplified ONNX': Path("simplified_temp.onnx"),
    }
    
    for name, path in models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {name}: no encontrado")
    
    # Check for any .kmodel files
    kmodels = list(Path(".").glob("*.kmodel"))
    if kmodels:
        for kmodel in kmodels:
            size_mb = kmodel.stat().st_size / (1024 * 1024)
            print(f"  ✅ KModel: {kmodel.name} ({size_mb:.1f} MB)")
    else:
        print("  ℹ️  KModel: ninguno encontrado (ejecuta train_kpu_depth.py primero)")


def show_help():
    """Show help information."""
    print("""
KPU Depth Estimation Project - Herramientas disponibles:

📋 Comandos principales:
  python main.py check          Verificar entorno y dependencias
  python main.py train          Entrenar modelo (requiere dataset)
  python main.py export         Exportar modelo a ONNX
  python main.py info           Mostrar información del proyecto

🚀 Ejemplos de uso:
  # Verificar que todo está listo
  python main.py check
  
  # Entrenar modelo (si dataset disponible)
  python train_kpu_depth.py
  
  # Usar Docker (recomendado)
  docker-compose up kpu-training

📁 Estructura del proyecto:
  data/          - Dataset NYU Depth V2
  models/        - Modelos exportados
  scripts/       - Scripts de ayuda
  results/       - Resultados de inferencia

📚 Documentación:
  README_NEW.md      - Documentación principal
  TRAINING_GUIDE.md  - Guía técnica detallada
  QUICKSTART.md      - Inicio rápido
  DATASET_CARD.md    - Información del dataset
    """)


def main():
    parser = argparse.ArgumentParser(
        description="KPU Depth Estimation Project - Main entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='check',
        choices=['check', 'train', 'export', 'info', 'help'],
        help='Comando a ejecutar (default: check)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'check':
        print("=" * 60)
        print("  KPU Depth Estimation - Verificación de entorno")
        print("=" * 60)
        
        env_ok = check_environment()
        dataset_available = check_dataset()
        check_models()
        
        print("\n" + "=" * 60)
        if env_ok:
            if dataset_available:
                print("✅ LISTO: Entorno y dataset disponibles")
                print("   Ejecuta: python train_kpu_depth.py")
            else:
                print("⚠️  PARCIAL: Entorno listo, dataset faltante")
                print("   Descarga el dataset o usa datos dummy")
        else:
            print("❌ PROBLEMAS: Faltan dependencias")
            print("   Ejecuta: pip install -e .")
        print("=" * 60)
        
    elif args.command == 'train':
        print("\n🎯 Iniciando entrenamiento...")
        print("Ejecuta: python train_kpu_depth.py")
        print("\nO usa Docker: docker-compose up kpu-training")
        
    elif args.command == 'export':
        print("\n📤 Exportando modelo...")
        print("El modelo se exporta automáticamente al final del entrenamiento.")
        print("Para exportar manualmente, ejecuta train_kpu_depth.py")
        
    elif args.command == 'info':
        show_help()
        
    elif args.command == 'help':
        show_help()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
