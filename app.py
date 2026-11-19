from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------- configuración
MODEL_DIR = Path(__file__).parent / "modelo_emociones_distilbert"
IDIOMA_MODELO = "inglés"

# Etiquetas del dataset dair-ai/emotion (orden original del notebook de clase).
# Si el modelo guardó nombres genéricos (LABEL_0...) o en inglés, se traducen con este mapa.
ETIQUETAS = {
    "LABEL_0": "tristeza", "LABEL_1": "alegría", "LABEL_2": "amor",
    "LABEL_3": "ira", "LABEL_4": "miedo", "LABEL_5": "sorpresa",
    "sadness": "tristeza", "joy": "alegría", "love": "amor",
    "anger": "ira", "fear": "miedo", "surprise": "sorpresa",
}

# Un color fijo por emoción: se usa en el gráfico y en la distribución del lote.
COLORES = {
    "Tristeza": "#3B6EA5", "Alegría": "#D99A00", "Amor": "#C2417A",
    "Ira": "#C0392B", "Miedo": "#6B4FA0", "Sorpresa": "#1F9E89",
}

# Métricas medidas con este modelo sobre el conjunto de prueba (2000 frases), ver README.
METRICAS_PRUEBA = {"Accuracy": "0,900", "F1-macro": "0,840"}

EJEMPLOS = {
    "Tristeza": "I feel so empty and alone since she left.",
    "Alegría": "I am so happy, today was the best day of my life!",
    "Amor": "I feel so loved and cared for when I am with you.",
    "Ira": "I am furious, they lied to me again.",
    "Miedo": "I am terrified of walking home alone at night.",
    "Sorpresa": "I was shocked when I opened the door and everyone was there.",
}

st.set_page_config(page_title="Detector de emociones", page_icon=":material/psychology:",
                   layout="centered")


# ---------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo(ruta: str):
    """Se ejecuta una sola vez; Streamlit reutiliza el pipeline en cada rerun."""
    return pipeline("text-classification", model=ruta)


def traducir(etiqueta: str) -> str:
    """Devuelve el nombre de la emoción en español y con mayúscula inicial."""
    return ETIQUETAS.get(etiqueta, etiqueta).capitalize()


def a_tabla(salida: list[dict]) -> pd.DataFrame:
    """Convierte la salida del pipeline (top_k=None) en una tabla ordenada en %."""
    df = pd.DataFrame(salida)
    df["Emoción"] = df["label"].map(traducir)
    df["Probabilidad (%)"] = (df["score"] * 100).round(2)
    return df[["Emoción", "Probabilidad (%)"]].sort_values(
        "Probabilidad (%)", ascending=False, ignore_index=True)


def clasificar(clf, textos: list[str]) -> list[pd.DataFrame]:
    # truncation=True corta las frases de más de 512 tokens para que el modelo no falle.
    salidas = clf(textos, top_k=None, truncation=True)
    # Con un texto, algunas versiones devuelven lista plana de dicts.
    if salidas and isinstance(salidas[0], dict):
        salidas = [salidas]
    return [a_tabla(s) for s in salidas]


def escala_colores():
    """Escala de Altair que asigna a cada emoción su color de COLORES."""
    return alt.Scale(domain=list(COLORES), range=list(COLORES.values()))


def grafico(df: pd.DataFrame):
    """Barras horizontales ordenadas; la emoción ganadora se ve opaca y el resto atenuado."""
    df = df.assign(Etiqueta=df["Probabilidad (%)"].map(lambda p: f"{p:.1f} %"))
    ganadora = df.iloc[0]["Emoción"]
    base = alt.Chart(df).encode(
        # La tabla ya viene ordenada de mayor a menor; se usa ese mismo orden en el eje.
        y=alt.Y("Emoción:N", sort=df["Emoción"].tolist(), title=None,
                axis=alt.Axis(labelFontSize=13, ticks=False, domain=False)),
        x=alt.X("Probabilidad (%):Q", scale=alt.Scale(domain=[0, 100]), title=None,
                axis=alt.Axis(tickCount=5, labelExpr="datum.value + ' %'")),
    )
    barras = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("Emoción:N", scale=escala_colores(), legend=None),
        opacity=alt.condition(alt.datum["Emoción"] == ganadora, alt.value(1.0), alt.value(0.4)),
        tooltip=["Emoción", "Probabilidad (%)"])
    texto = base.mark_text(align="left", dx=6, fontSize=12, color="#334155").encode(text="Etiqueta:N")
    return (barras + texto).properties(height=250).configure_view(stroke=None)


def grafico_distribucion(res: pd.DataFrame):
    """Cuántas frases del lote cayeron en cada emoción."""
    conteo = res["Emoción"].value_counts().rename_axis("Emoción").reset_index(name="Frases")
    return alt.Chart(conteo).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=alt.Y("Emoción:N", sort="-x", title=None, axis=alt.Axis(ticks=False, domain=False)),
        x=alt.X("Frases:Q", title="Número de frases", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("Emoción:N", scale=escala_colores(), legend=None),
        tooltip=["Emoción", "Frases"],
    ).properties(height=40 * len(conteo) + 30).configure_view(stroke=None)


# Muestra la confianza como barra de progreso dentro de las tablas.
COLUMNA_CONFIANZA = st.column_config.ProgressColumn(
    "Confianza", min_value=0, max_value=100, format="%.1f %%")

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


# ---------------------------------------------------------------- encabezado
st.title("Detector de emociones")
st.caption("Clasificación de frases en seis emociones con un modelo DistilBERT ajustado por el equipo.")
st.info(f"**Idioma del modelo: {IDIOMA_MODELO}.** Fue entrenado con frases en {IDIOMA_MODELO}; "
        "con otros idiomas los resultados no son confiables.", icon=":material/translate:")

# Validación: carpeta del modelo (debe existir y contener config.json)
if not (MODEL_DIR / "config.json").is_file():
    st.error(f"No se encontró la carpeta del modelo `{MODEL_DIR.name}` (o está vacía). "
             "Colócala junto a `app.py` con config.json, pesos y tokenizador (ver README) "
             "y recarga la página.", icon=":material/error:")
    st.stop()

try:
    clf = cargar_modelo(str(MODEL_DIR))
except Exception as e:  # carpeta existe pero está incompleta/corrupta
    st.error(f"No se pudo cargar el modelo: {e}", icon=":material/error:")
    st.stop()

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Configuración")
    umbral = st.slider("Umbral de confianza (%)", 0, 100, 50,
                       help="Si la probabilidad de la emoción principal es menor, "
                            "la app avisa que el modelo no está seguro.")
    st.divider()
    st.subheader("Modelo")
    st.markdown(f"**Base:** `distilbert-base-uncased`  \n"
                "**Ajuste fino:** `dair-ai/emotion`  \n"
                f"**Idioma:** {IDIOMA_MODELO}  \n"
                f"**Carpeta:** `{MODEL_DIR.name}`")
    st.caption("Resultados en el conjunto de prueba (2000 frases)")
    for col, (nombre, valor) in zip(st.columns(2), METRICAS_PRUEBA.items()):
        col.metric(nombre, valor)

st.session_state.setdefault("historial", [])

tab1, tab2, tab3 = st.tabs([":material/edit_note: Una frase",
                            ":material/table_view: Análisis por lotes",
                            ":material/history: Historial"])

# ---------------------------------------------------------------- una frase
with tab1:
    with st.container(border=True):
        st.markdown("**Frases de ejemplo**")
        cols = st.columns(3)
        for i, (nombre, frase) in enumerate(EJEMPLOS.items()):
            cols[i % 3].button(nombre, key=f"ej{i}", on_click=cargar_ejemplo, args=(frase,),
                               width="stretch")

        texto = st.text_area("Frase a analizar", key="texto", height=100,
                             placeholder="Ej.: I can't believe how happy I am today")
        analizar = st.button("Analizar", type="primary", icon=":material/search:")

    if analizar:
        if not texto.strip():
            st.warning("Escribe una frase antes de analizar.", icon=":material/edit:")
        else:
            if parece_espanol(texto):
                st.warning("La frase parece estar en español. El modelo fue entrenado en "
                           f"{IDIOMA_MODELO}, así que el resultado puede no ser confiable.",
                           icon=":material/translate:")
            tabla = clasificar(clf, [texto])[0]
            top, segunda = tabla.iloc[0], tabla.iloc[1]

            with st.container(border=True):
                st.markdown("**Resultado**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Emoción detectada", top["Emoción"])
                c2.metric("Confianza", f"{top['Probabilidad (%)']:.1f} %")
                c3.metric(f"Segunda opción ({segunda['Probabilidad (%)']:.1f} %)",
                          segunda["Emoción"])
                if top["Probabilidad (%)"] < umbral:
                    st.warning(f"Confianza baja (menor a {umbral} %). "
                               "Interpreta este resultado con cautela.", icon=":material/warning:")
                st.markdown("**Probabilidad de cada emoción**")
                st.altair_chart(grafico(tabla), width="stretch")

            st.session_state["historial"].append({
                "Frase": texto.strip(), "Emoción": top["Emoción"],
                "Confianza (%)": float(top["Probabilidad (%)"])})

# ---------------------------------------------------------------- lotes
with tab2:
    with st.container(border=True):
        st.markdown("**Varias frases a la vez**: una por línea, o un CSV con una columna de texto.")
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
                    st.error(f"No se pudo leer el CSV: {e}", icon=":material/error:")
        analizar_lote = st.button("Analizar lote", type="primary", icon=":material/play_arrow:")

    if analizar_lote:
        if not frases:
            st.warning("No hay frases para analizar.", icon=":material/edit:")
        else:
            with st.spinner(f"Analizando {len(frases)} frases..."):
                tablas = clasificar(clf, frases)
            res = pd.DataFrame({
                "Frase": frases,
                "Emoción": [t.iloc[0]["Emoción"] for t in tablas],
                "Confianza (%)": [t.iloc[0]["Probabilidad (%)"] for t in tablas]})
            res["Segura"] = (res["Confianza (%)"] >= umbral).map({True: "sí", False: "no"})

            m1, m2, m3 = st.columns(3)
            m1.metric("Frases analizadas", len(res))
            m2.metric("Confianza media", f"{res['Confianza (%)'].mean():.1f} %")
            m3.metric("Bajo el umbral", int((res["Segura"] == "no").sum()))

            st.dataframe(res, width="stretch", hide_index=True,
                         column_config={"Confianza (%)": COLUMNA_CONFIANZA})
            st.markdown("**Distribución de emociones**")
            st.altair_chart(grafico_distribucion(res), width="stretch")
            st.download_button("Descargar resultados (CSV)",
                               res.to_csv(index=False).encode("utf-8-sig"),
                               "resultados_emociones.csv", "text/csv",
                               icon=":material/download:")

# ---------------------------------------------------------------- historial
with tab3:
    if st.session_state["historial"]:
        hist = pd.DataFrame(st.session_state["historial"])
        st.dataframe(hist, width="stretch", hide_index=True,
                     column_config={"Confianza (%)": COLUMNA_CONFIANZA})
        b1, b2 = st.columns(2)
        b1.download_button("Descargar historial (CSV)",
                           hist.to_csv(index=False).encode("utf-8-sig"),
                           "historial_emociones.csv", "text/csv",
                           icon=":material/download:", width="stretch")
        if b2.button("Borrar historial", icon=":material/delete:", width="stretch"):
            st.session_state["historial"] = []
            st.rerun()
    else:
        st.caption("Aún no has analizado frases en esta sesión.")
