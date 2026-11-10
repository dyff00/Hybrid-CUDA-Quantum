"""Chronos-2 zero-shot em toda a série, a cada 1 h. Rode no ambiente principal, com GPU."""
import time, torch
from chronos import Chronos2Pipeline
from comum import *
idx, y = carregar_serie(verboso=True); n = len(y)
ultimo = n - H
origens = np.arange(ultimo % PASSO_HIST, ultimo + 1, PASSO_HIST); origens = origens[origens >= CTX]
print(f"Janelas: {len(origens)}")
disp = "cuda" if torch.cuda.is_available() else "cpu"
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=disp)
print("Dispositivo:", disp)
Q = np.zeros((len(origens), H, 9)); t0 = time.time(); LOTE = 128
for a in range(0, len(origens), LOTE):
    bloco = origens[a:a + LOTE]; linhas = []
    for o in bloco:
        linhas.append(pd.DataFrame({"item_id": int(o), "timestamp": idx[o - CTX:o], "target": y[o - CTX:o]}))
    df = pd.concat(linhas, ignore_index=True)
    r = pipe.predict_df(df, id_column="item_id", timestamp_column="timestamp", target="target",
                        prediction_length=H, quantile_levels=QUANTIS, batch_size=LOTE, freq="30min")
    cols = [str(q) for q in QUANTIS]
    for j, o in enumerate(bloco):
        p = r[r["item_id"] == int(o)].sort_values("timestamp")
        if len(p) != H:
            raise SystemExit(f"Origem {o}: vieram {len(p)} passos em vez de {H}.")
        Q[a + j] = p[cols].values
    print(f"{min(a + LOTE, len(origens))}/{len(origens)} em {time.time() - t0:.0f} s", flush=True)
Y = np.stack([y[o:o + H] for o in origens])
np.savez_compressed(SAIDA / "historico_chronos.npz", origens=origens, Q=Q, Y=Y, n=n, serie=y,
                    tempos=np.array([str(t) for t in idx]))
hist = carregar_historico(); _, i_te, _ = divisao(hist)
res = resumo(metricas(Y[i_te], Q[i_te]))
print("\nTeste de sanidade contra a fase 1")
print(f"{'métrica':10s} {'agora':>8s} {'fase 1':>8s}")
ok = True
for k in FASE1:
    print(f"{k:10s} {res[k]:8.3f} {FASE1[k]:8.3f}")
    ok &= abs(res[k] - FASE1[k]) / FASE1[k] < 0.05
print("OK: reproduz a fase 1." if ok else "ATENÇÃO: diferença acima de 5%. Me mande esta saída antes de seguir.")
salvar("r0_sanidade.json", {"agora": res, "fase1": FASE1, "ok": bool(ok), "janelas_total": len(origens), "dispositivo": disp, "segundos": time.time() - t0})
