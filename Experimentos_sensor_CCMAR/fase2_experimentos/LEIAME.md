# Fase 2 – Experimentos E0 a E5 sobre o erro do Chronos-2

Seis experimentos sobre o sensor CCMAR, rodados em 10/11/2026 em um computador com Linux e
GPU RTX 5070. Os resultados estão em [../resultados/](../resultados/).

| Script | Exp. | O que faz |
|---|---|---|
| `00_verificar.py` | – | Confere a série e se o período de teste é o mesmo da fase 1 |
| `01_historico_chronos.py` | E0 | Chronos-2 zero-shot em todo o histórico, uma previsão por hora (GPU) |
| `e1_onde_erra.py` | E1 | Espectro do erro do Chronos-2 por banda, em 1, 6, 12 e 24 h |
| `e2_mare.py` | E2 | Correção de maré harmônica (4 e 9 constituintes) |
| `e3_bayes.py` | E3 | Erro bayesiano e calibração conformal do intervalo p10–p90 |
| `e4_circuito.py` | E4 | Circuitos CUDA-Q (maré e genérico) contra KAN, MLP e harmônica |
| `e5_gpu.py` | E5 | Tempo do circuito em CUDA-Q (GPU e CPU) e PennyLane, de 4 a 26 qubits |
| `06_empacotar.py` | – | Junta as saídas em `saida/resultados_para_claude.zip` |
| `comum.py`, `estat.py` | – | Funções compartilhadas e testes pareados (Wilcoxon, Holm, bootstrap) |

## 1. Conferir os dados (1 min)

Precisa do ambiente `.venv-principal` da fase 1 (veja `../fase1_replicacao/README.md`) e do
CSV em `../dados/sensor_1_CCMAR.csv`.

```bash
cd Experimentos_sensor_CCMAR
source .venv-principal/bin/activate
cd fase2_experimentos
python 00_verificar.py
```

Deve aparecer "OK: período de teste igual ao da fase 1".

## 2. Chronos-2 em todo o histórico (E0, GPU)

```bash
python 01_historico_chronos.py
deactivate
```

Deve aparecer "OK: reproduz a fase 1" (MAE perto de 4,62 cm). Na RTX 5070 leva cerca de 3 s.

## 3. Criar o ambiente quântico (só uma vez)

```bash
cd ..
uv venv .venv-quantico --python 3.12
source .venv-quantico/bin/activate
uv pip install -r fase2_experimentos/requirements-quantico.txt
python -c "import cudaq, jax, pennylane; print('ok', cudaq.__version__)"
```

## 4. Rodar E1 a E5

```bash
cd fase2_experimentos
bash rodar_quantico.sh
```

E1 a E3 levam menos de 1 min. O treino dos 6 circuitos de E4 roda em CPU, cerca de 20 min
cada; para escolher quantos rodam ao mesmo tempo, use `PROC=4 bash rodar_quantico.sh`. Se
parar no meio, rode de novo: os circuitos já treinados são aproveitados. E5 mede o alvo
`nvidia` por último.

As saídas ficam em `saida/` (JSON, CSV e logs de cada experimento).
