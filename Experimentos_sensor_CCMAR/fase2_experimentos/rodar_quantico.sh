#!/usr/bin/env bash
# Comparações 1 a 4 e medição de GPU. Rode dentro do ambiente .venv-quantico.
set -euo pipefail
cd "$(dirname "$0")"
PROC=${PROC:-$(( $(nproc) / 2 > 1 ? $(nproc) / 2 : 1 ))}
python e1_onde_erra.py | tee saida/log_e1.txt
python e2_mare.py      | tee saida/log_e2.txt
python e3_bayes.py     | tee saida/log_e3.txt
echo "Treinando 6 circuitos, $PROC por vez..."
printf 'mare 0\ngen 0\nmare 1\ngen 1\nmare 2\ngen 2\n' | OMP_NUM_THREADS=1 xargs -P "$PROC" -L1 python e4_circuito.py treinar | tee saida/log_e4_treino.txt
python e4_circuito.py resumir | tee saida/log_e4.txt
python e5_gpu.py | tee saida/log_e5.txt
python 06_empacotar.py
