#!/bin/bash

# 1. Cargar las funciones de Conda desde tu ruta de miniconda
source /lhome/ext/uovi123/uovi123l/miniconda3/etc/profile.d/conda.sh

# 2. Activar tu entorno específico
conda activate snn_hgcal

# 2. Ir al directorio donde está el código
cd /lhome/ext/uovi123/uovi123l/SNN-Cal/SNN

python $1