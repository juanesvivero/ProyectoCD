from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------- configuración
MODEL_DIR = Path(__file__).parent / "modelo_emociones_distilbert"
IDIOMA_MODELO = "inglés"

# Etiquetas del dataset dair-ai/emotion (orden original del notebook de clase).
# Si el modelo guardó nombres genéricos (LABEL_0...), se traducen con este mapa.
ETIQUETAS = {
    "LABEL_0": "tristeza", "LABEL_1": "alegría", "LABEL_2": "amor",
    "LABEL_3": "ira", "LABEL_4": "miedo", "LABEL_5": "sorpresa",
    "sadness": "tristeza", "joy": "alegría", "love": "amor",
    "anger": "ira", "fear": "miedo", "surprise": "sorpresa",
}
EMOJIS = {"tristeza": "😢", "alegría": "😄", "amor": "❤️",
          "ira": "😠", "miedo": "😨", "sorpresa": "😲"}

EJEMPLOS = {
    "😢 Tristeza": "I feel so empty and alone since she left.",
    "😄 Alegría": "I am so happy, today was the best day of my life!",
    "❤️ Amor": "I feel so loved and cared for when I am with you.",
    "😠 Ira": "I am furious, they lied to me again.",
    "😨 Miedo": "I am terrified of walking home alone at night.",
    "😲 Sorpresa": "I was shocked when I opened the door and everyone was there.",
}

st.set_page_config(page_title="Detector de emociones", page_icon="🎭", layout="centered")


# ---------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo(ruta: str):
    """Se ejecuta una sola vez; Streamlit reutiliza el pipeline en cada rerun."""
    return pipeline("text-classification", model=ruta)


def traducir(etiqueta: str) -> str:
    return ETIQUETAS.get(etiqueta, etiqueta)


def a_tabla(salida: list[dict]) -> pd.DataFrame:
    """Convierte la salida del pipeline (top_k=None) en una tabla ordenada en %."""
    df = pd.DataFrame(salida)
    df["Emoción"] = df["label"].map(traducir)
    df["Probabilidad (%)"] = (df["score"] * 100).round(2)
    return df[["Emoción", "Probabilidad (%)"]].sort_values(
        "Probabilidad (%)", ascending=False, ignore_index=True)


def clasificar(clf, textos: list[str]) -> list[pd.DataFrame]:
    salidas = clf(textos, top_k=None, truncation=True)
    # Con un texto, algunas versiones devuelven lista plana de dicts.
    if salidas and isinstance(salidas[0], dict):
        salidas = [salidas]
    return [a_tabla(s) for s in salidas]


def grafico(df: pd.DataFrame):
    base = alt.Chart(df).encode(
        y=alt.Y("Emoción:N", sort="-x", title=None),
        x=alt.X("Probabilidad (%):Q", scale=alt.Scale(domain=[0, 100]), title="Probabilidad (%)"),
    )
    barras = base.mark_bar().encode(
        tooltip=["Emoción", "Probabilidad (%)"],
        color=alt.condition(alt.datum["Emoción"] == df.iloc[0]["Emoción"],
                            alt.value("#e4572e"), alt.value("#9aa5b1")))
    texto = base.mark_text(align="left", dx=4).encode(text=alt.Text("Probabilidad (%):Q", format=".1f"))
    return (barras + texto).properties(height=260)


PALABRAS_ES = {"el", "la", "los", "las", "un", "una", "que", "y", "de", "es", "en", "por", "con",
               "para", "pero", "mi", "yo", "me", "estoy", "esta", "está", "muy", "no", "se", "su",
               "como", "qué", "cuando", "ahora", "siento", "tengo", "soy", "eres", "te"}


def parece_espanol(texto: str) -> bool:
    """Heurística simple: signos/letras propios del español o varias palabras comunes."""
    t = texto.lower()
    if any(c in t for c in "ñ¿¡áéíóú"):
        return True
    palabras = t.replace(",", " ").replace(".", " ").split()
    return sum(p in PALABRAS_ES for p in palabras) >= 2


def cargar_ejemplo(frase: str):
    st.session_state["texto"] = frase


# ---------------------------------------------------------------- interfaz
st.title("🎭 Detector de emociones")
st.caption("Escribe una frase y el modelo dirá qué emoción expresa: "
           "tristeza, alegría, amor, ira, miedo o sorpresa.")
st.info(f"🌐 **Idioma del modelo: {IDIOMA_MODELO}.** Fue entrenado con frases en {IDIOMA_MODELO}; "
        "con otros idiomas los resultados no son confiables.")

# Validación: carpeta del modelo (debe existir y contener config.json)
if not (MODEL_DIR / "config.json").is_file():
    st.error(f"No se encontró la carpeta del modelo `{MODEL_DIR.name}` (o está vacía). "
             "Colócala junto a `app.py` con config.json, pesos y tokenizador (ver README) "
             "y recarga la página.")
    st.stop()

try:
    clf = cargar_modelo(str(MODEL_DIR))
except Exception as e:  # carpeta existe pero está incompleta/corrupta
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

st.sidebar.header("Configuración")
umbral = st.sidebar.slider("Umbral de confianza (%)", 0, 100, 50,
                           help="Si la probabilidad de la emoción principal es menor, "
                                "la app avisa que el modelo no está seguro.")
with st.sidebar.expander("Sobre el modelo", expanded=True):
    st.markdown(f"- **Base:** DistilBERT (`distilbert-base-uncased`)\n"
                "- **Ajuste fino:** dataset `dair-ai/emotion`\n"
                "- **Emociones:** tristeza, alegría, amor, ira, miedo, sorpresa\n"
                f"- **Idioma:** {IDIOMA_MODELO}\n"
                f"- **Ruta:** `{MODEL_DIR.name}`")
st.session_state.setdefault("historial", [])

tab1, tab2, tab3 = st.tabs(["Una frase", "Análisis por lotes", "Historial"])

# ---- Una frase
with tab1:
    st.write("**Frases de ejemplo:**")
    cols = st.columns(3)
    for i, (nombre, frase) in enumerate(EJEMPLOS.items()):
        cols[i % 3].button(nombre, key=f"ej{i}", on_click=cargar_ejemplo, args=(frase,),
                           width="stretch")

    texto = st.text_area("Frase a analizar", key="texto", height=100,
                         placeholder="Ej.: I can't believe how happy I am today")
    if st.button("Analizar", type="primary"):
        if not texto.strip():
            st.warning("Escribe una frase antes de analizar.")
        else:
            if parece_espanol(texto):
                st.warning("La frase parece estar en español. El modelo fue entrenado en "
                           f"{IDIOMA_MODELO}, así que el resultado puede no ser confiable.")
            tabla = clasificar(clf, [texto])[0]
            top = tabla.iloc[0]
            st.subheader(f"{EMOJIS.get(top['Emoción'], '')} {top['Emoción'].capitalize()}")
            st.metric("Confianza", f"{top['Probabilidad (%)']:.1f} %")
            if top["Probabilidad (%)"] < umbral:
                st.warning(f"⚠️ Confianza baja (menor a {umbral} %). "
                           "Interpreta este resultado con cautela.")
            st.altair_chart(grafico(tabla), width="stretch")
            st.session_state["historial"].append({
                "Frase": texto.strip(), "Emoción": top["Emoción"],
                "Confianza (%)": float(top["Probabilidad (%)"])})

# ---- Lotes
with tab2:
    st.write("Una frase por línea, o sube un CSV con una columna de texto.")
    modo = st.radio("Origen", ["Escribir líneas", "Archivo CSV"], horizontal=True)
    frases: list[str] = []
    if modo == "Escribir líneas":
        bloque = st.text_area("Frases (una por línea)", height=150, key="lote")
        frases = [l.strip() for l in bloque.splitlines() if l.strip()]
    else:
        archivo = st.file_uploader("CSV", type="csv")
        if archivo is not None:
            try:
                datos = pd.read_csv(archivo)
                col = st.selectbox("Columna con el texto", datos.columns)
                frases = datos[col].dropna().astype(str).str.strip().tolist()
                frases = [f for f in frases if f]
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
    if st.button("Analizar lote"):
        if not frases:
            st.warning("No hay frases para analizar.")
        else:
            with st.spinner(f"Analizando {len(frases)} frases..."):
                tablas = clasificar(clf, frases)
            res = pd.DataFrame({
                "Frase": frases,
                "Emoción": [t.iloc[0]["Emoción"] for t in tablas],
                "Confianza (%)": [t.iloc[0]["Probabilidad (%)"] for t in tablas]})
            res["Segura"] = (res["Confianza (%)"] >= umbral).map({True: "sí", False: "no"})
            st.dataframe(res, width="stretch")
            st.subheader("Distribución de emociones")
            st.bar_chart(res["Emoción"].value_counts())
            st.download_button("⬇️ Descargar resultados (CSV)",
                               res.to_csv(index=False).encode("utf-8-sig"),
                               "resultados_emociones.csv", "text/csv")

# ---- Historial
with tab3:
    if st.session_state["historial"]:
        st.dataframe(pd.DataFrame(st.session_state["historial"]), width="stretch")
        if st.button("Borrar historial"):
            st.session_state["historial"] = []
            st.rerun()
    else:
        st.write("Aún no has analizado frases en esta sesión.")
