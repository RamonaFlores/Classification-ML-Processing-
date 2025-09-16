#!/usr/bin/env python3
"""
DataPros - Script de Ejecución Principal
========================================

Este script ejecuta el análisis completo de clasificación de ingresos adultos.

Uso:
    python run_analysis.py

Autor: DataPros Team
Fecha: 2024
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """
    Verifica que se cumplan los requisitos del sistema
    """
    print("🔍 Verificando requisitos del sistema...")
    
    # Verificar Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Verificar Java (requerido para Spark)
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Java está instalado")
        else:
            print("❌ Error: Java no está instalado (requerido para Spark)")
            return False
    except FileNotFoundError:
        print("❌ Error: Java no está instalado (requerido para Spark)")
        return False
    
    return True

def install_dependencies():
    """
    Instala las dependencias necesarias
    """
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al instalar dependencias")
        return False

def run_main_script():
    """
    Ejecuta el script principal de clasificación
    """
    print("\n🚀 Ejecutando análisis de clasificación...")
    try:
        script_path = Path("src/adult_income_classification.py")
        if not script_path.exists():
            print(f"❌ Error: No se encontró el archivo {script_path}")
            return False
        
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("✅ Análisis completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar el análisis: {e}")
        return False

def run_demo():
    """
    Ejecuta el script de demostración
    """
    print("\n🎯 Ejecutando demostración de predicciones...")
    try:
        script_path = Path("src/demo_predictions.py")
        if not script_path.exists():
            print(f"❌ Error: No se encontró el archivo {script_path}")
            return False
        
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("✅ Demostración completada exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar la demostración: {e}")
        return False

def main():
    """
    Función principal
    """
    print("=" * 60)
    print("🎯 DataPros - Clasificación de Ingresos Adultos")
    print("=" * 60)
    
    # Verificar requisitos
    if not check_requirements():
        print("\n❌ Los requisitos del sistema no se cumplen")
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        print("\n❌ Error al instalar dependencias")
        sys.exit(1)
    
    # Ejecutar análisis principal
    if not run_main_script():
        print("\n❌ Error en el análisis principal")
        sys.exit(1)
    
    # Ejecutar demostración
    if not run_demo():
        print("\n❌ Error en la demostración")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ¡Análisis completado exitosamente!")
    print("=" * 60)
    print("\n📋 Archivos generados:")
    print("   • models/adult_income_model/ - Modelo entrenado")
    print("   • Logs de ejecución en consola")
    
    print("\n💡 Próximos pasos:")
    print("   • Revisar el notebook: src/notebook_analysis.ipynb")
    print("   • Usar el modelo para nuevas predicciones")
    print("   • Considerar optimizaciones adicionales")

if __name__ == "__main__":
    main()
