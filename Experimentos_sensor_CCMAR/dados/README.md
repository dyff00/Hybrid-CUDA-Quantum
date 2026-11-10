# Dados

Coloque aqui o arquivo `sensor_1_CCMAR.csv`, exportado da API do sensor CCMAR, com as colunas
`data_hora` e `valor` (nível em cm).

A série usada nos experimentos de 09 e 10/11/2026 tem 3.814 leituras, de 08/08/2026 01:30 a
26/10/2026 14:30, com 5 lacunas preenchidas por interpolação linear.

Os scripts da fase 1 (`../fase1_replicacao/config.py`) e da fase 2
(`../fase2_experimentos/comum.py`) leem o arquivo desta pasta. O `01_preparar_dados.py` da
fase 1 grava aqui também a série em grade de 30 min (`serie_30min.csv`).
