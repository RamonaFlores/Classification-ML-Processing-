#!/bin/bash
# Script para activar el entorno virtual del proyecto DataPros

echo "🚀 Activando entorno virtual para DataPros Adult Income Classification..."
echo "=" * 60

# Verificar si el entorno virtual existe
if [ ! -d ".venv" ]; then
    echo "❌ Error: El entorno virtual no existe."
    echo "💡 Ejecuta primero: python3 -m venv .venv"
    exit 1
fi

# Activar el entorno virtual
source .venv/bin/activate

# Verificar que se activó correctamente
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Entorno virtual activado: $VIRTUAL_ENV"
    echo "🐍 Python version: $(python --version)"
    echo "📦 Pip version: $(pip --version)"
    echo ""
    echo "🎯 Comandos disponibles:"
    echo "   • python src/adult_income_classification.py  # Ejecutar análisis principal"
    echo "   • python src/demo_predictions.py            # Ejecutar demostración"
    echo "   • jupyter notebook src/notebook_analysis.ipynb  # Abrir notebook"
    echo "   • python run_analysis.py                    # Ejecutar todo automáticamente"
    echo ""
    echo "💡 Para desactivar el entorno: deactivate"
    echo "=" * 60
else
    echo "❌ Error: No se pudo activar el entorno virtual"
    exit 1
fi
