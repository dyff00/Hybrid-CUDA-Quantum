"""Junta tudo para me enviar."""
import zipfile
from pathlib import Path
s = Path(__file__).resolve().parent / "saida"; d = s / "resultados_para_claude.zip"
fs = [f for f in sorted(s.iterdir()) if f.suffix in (".json", ".csv", ".txt")]
with zipfile.ZipFile(d, "w", zipfile.ZIP_DEFLATED) as z:
    for f in fs: z.write(f, f.name)
print("Arquivos:", [f.name for f in fs]); print("Pronto:", d)
