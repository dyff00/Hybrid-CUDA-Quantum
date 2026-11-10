python3 e1_espectro.py > log1.txt 2>&1
printf 'mare 0\ngen 0\nmare 1\ngen 1\nmare 2\ngen 2\n' | xargs -P2 -L1 python3 e2_q.py > log2.txt 2>&1
python3 e4_tiros.py > log4.txt 2>&1
python3 e6_substituto.py > log6.txt 2>&1
python3 e5_tempo.py > log5.txt 2>&1
echo fim
