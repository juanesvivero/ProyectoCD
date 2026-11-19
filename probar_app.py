"""Prueba automática de app.py con tu modelo real. Uso: python probar_app.py app.py (requiere streamlit>=1.28)."""
import sys
from streamlit.testing.v1 import AppTest
APP = sys.argv[1]
def run(label, fn):
    at = AppTest.from_file(APP, default_timeout=120).run()
    fn(at); print("PASS" if not at.exception else "EXC", label, [e.value for e in at.exception])
    return at

at = AppTest.from_file(APP, default_timeout=120).run()
assert not at.exception, at.exception
print("carga inicial OK; tabs:", [t.label for t in at.tabs])

# 1) frase vacía
at.button[[b.label for b in at.button].index("Analizar")].click().run()
print("vacía ->", [w.value for w in at.warning])

# 2) frase en inglés
at.text_area(key="texto").set_value("I feel so happy today").run()
at.button[[b.label for b in at.button].index("Analizar")].click().run()
print("inglés -> exc:", at.exception, "| subheader:", [s.value for s in at.subheader], "| metric:", [m.value for m in at.metric], "| warnings:", [w.value for w in at.warning])
print("historial:", at.session_state["historial"])

# 3) español
at.text_area(key="texto").set_value("Ahora que pasa cuando una frase esta en español?").run()
at.button[[b.label for b in at.button].index("Analizar")].click().run()
print("español -> warnings:", [w.value for w in at.warning])

# 4) ejemplo
at.button(key="ej0").click().run()
print("ejemplo ->", at.text_area(key="texto").value)

# 5) lote
at.tabs[1]  # pestaña lotes
at.text_area(key="lote").set_value("I feel so happy\nI am not sad at all\nI love you").run()
at.button[[b.label for b in at.button].index("Analizar lote")].click().run()
print("lote -> exc:", at.exception, "| dataframes:", [d.value.to_dict("records") for d in at.dataframe])
