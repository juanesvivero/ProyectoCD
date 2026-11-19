# Detector de emociones con DistilBERT + Streamlit

App web hecha con Streamlit: se escribe una frase en **inglés** y el modelo dice qué emoción expresa (tristeza, alegría, amor, ira, miedo o sorpresa), qué tan seguro está y cuánta probabilidad le da a cada una de las seis.

**Integrantes:** Carlos Ortega y Juan Esteban Vivero
**Materia:** Análisis, Procesamiento y Aplicación de IA — clase «Fine-tuning y evaluación de modelos de lenguaje»

![Pantalla principal](capturas/01_pantalla_principal.png)

## Modelo

- **Base:** `distilbert-base-uncased`, que ajustamos nosotros en el notebook de clase (Bloque 4.4) con el dataset `dair-ai/emotion` (tuits cortos en inglés).
- **Entrenamiento:** 4000 frases, 3 épocas, `learning_rate=2e-5`, `max_length=64` y semilla 42 (son los mismos valores de `entrenar_modelo.py`).
- **Etiquetas** (`id2label` en `config.json`): 0 tristeza, 1 alegría, 2 amor, 3 ira, 4 miedo, 5 sorpresa.
- **Idioma: inglés.** La app lo avisa arriba y, si la frase parece estar en español, muestra otra advertencia.
- **Resultados en el conjunto de prueba** (2000 frases, con este mismo modelo): **accuracy 0,900** y **F1-macro 0,840**.

| Emoción | Precisión | Recall | F1 | Frases en prueba |
|---|---|---|---|---|
| tristeza | 0,933 | 0,957 | 0,945 | 581 |
| alegría | 0,901 | 0,944 | 0,922 | 695 |
| amor | 0,794 | 0,704 | 0,747 | 159 |
| ira | 0,922 | 0,862 | 0,891 | 275 |
| miedo | 0,871 | 0,906 | 0,888 | 224 |
| sorpresa | 0,800 | 0,545 | 0,649 | 66 |

- La carpeta `modelo_emociones_distilbert/` pesa unos 256 MB y **no está en GitHub**, porque GitHub no deja subir archivos de más de 100 MB. Por eso está en el `.gitignore`.
- Si no tienes la carpeta, puedes generarla con `entrenar_modelo.py`, que repite el Bloque 4 del notebook (en Colab con GPU demora unos minutos).

## Instalación y ejecución

Necesitas Python 3.10 o más nuevo. Lo probamos con Python 3.14, Streamlit 1.67, Transformers 5.22 y PyTorch 2.15.

```bash
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>
pip install -r requirements.txt
```

Después copia la carpeta `modelo_emociones_distilbert/` al lado de `app.py` (con `config.json`, `model.safetensors`, `tokenizer.json` y `tokenizer_config.json` adentro) y ejecuta:

```bash
streamlit run app.py
```

Se abre sola en `http://localhost:8501`.

Opcional: `python probar_app.py app.py` prueba la app sin abrir el navegador (frase vacía, una en inglés, una en español, un ejemplo y un lote) e imprime lo que pasa.

Archivos del proyecto:

```
proyecto/
├── app.py                         la app
├── requirements.txt               dependencias
├── README.md
├── .gitignore                     excluye la carpeta del modelo
├── .streamlit/config.toml         colores y tema
├── entrenar_modelo.py             entrena el modelo si no lo tienes
├── probar_app.py                  prueba rápida de app.py
├── capturas/                      capturas del README
└── modelo_emociones_distilbert/   (no se sube a GitHub)
```

## Funcionalidades

**Obligatorias**
- Campo de texto y botón **Analizar**.
- Muestra la emoción detectada, la confianza en porcentaje y la segunda emoción más probable.
- Gráfico de barras con las seis emociones, de mayor a menor y con el porcentaje en cada barra. Cada emoción tiene su color y la ganadora se resalta.
- El modelo se carga una sola vez con `@st.cache_resource`.
- Validaciones: avisa si la frase está vacía (o solo tiene espacios) y muestra un error si la carpeta del modelo no existe o está vacía. Si el modelo está dañado, también muestra el error sin que se caiga la app.
- Frases largas: usa `truncation=True`, que corta el texto a 512 tokens. Lo probamos con una frase de unas 3000 palabras y no falla.
- Aviso del idioma del modelo (inglés) y una advertencia extra si la frase parece estar en español.

**Adicionales (hicimos 4)**
1. **Umbral de confianza:** slider en la barra lateral (50 % por defecto). Si la emoción principal queda por debajo, sale una advertencia.
2. **Análisis por lotes:** una frase por línea o un archivo CSV (se elige la columna con el texto). Muestra un resumen (frases analizadas, confianza media y cuántas quedan bajo el umbral), una tabla, un gráfico de distribución y un botón para descargar los resultados en CSV.
3. **Historial:** guarda las frases analizadas en la sesión con `st.session_state`. Se puede descargar en CSV o borrar.
4. **Frases de ejemplo:** seis botones, uno por emoción, que ponen una frase de prueba en el campo de texto.

## Capturas de pantalla

| | |
|---|---|
| **Resultado con gráfico** (ejemplo de sorpresa) ![Resultado](capturas/02_resultado_grafico.png) | **Aviso de baja confianza** (41,1 %, menor al umbral de 50 %) ![Baja confianza](capturas/03_baja_confianza.png) |
| **Aviso de frase en español** ![Español](capturas/04_aviso_espanol.png) | **Análisis por lotes** ![Lote](capturas/05_lote.png) |
| **Historial de la sesión** ![Historial](capturas/06_historial.png) | **Error si falta la carpeta del modelo** ![Error](capturas/07_error_carpeta_ausente.png) |

## Análisis de errores

En el conjunto de prueba el modelo acierta el 90 % de las frases, pero el F1-macro es 0,840 porque le va peor con las emociones que tienen pocos ejemplos. Todos los números de esta sección salen de correr nuestro modelo (se muestran la emoción principal y las dos siguientes).

| Frase | Predicción (top 3) | ¿Correcto? |
|---|---|---|
| I am not sad at all | tristeza 96,9 %, alegría 0,9 %, miedo 0,7 % | ✗ negación |
| I don't hate you | ira 59,6 %, tristeza 34,0 %, amor 3,0 % | ✗ negación |
| Oh great, another Monday. I'm just thrilled | alegría 93,5 %, sorpresa 2,9 %, amor 1,6 % | ✗ sarcasmo |
| I can't believe what just happened | miedo 41,1 %, sorpresa 35,9 %, alegría 7,8 % | dudoso (debería ser sorpresa) |
| I'm happy for her but I will miss her terribly | alegría 60,1 %, tristeza 30,4 %, amor 6,4 % | parcial (mezcla) |
| I can't believe it, I won the lottery! | alegría 85,3 %, sorpresa 4,1 %, ira 3,0 % | parcial (falta sorpresa) |
| Estoy muy feliz porque aprobé el examen | alegría 87,2 %, tristeza 2,8 %, miedo 2,8 % | ✓ por casualidad |
| Tengo mucho miedo de quedarme solo | alegría 73,6 %, amor 10,4 %, tristeza 5,4 % | ✗ español |
| Te odio, me mentiste otra vez | alegría 67,8 %, amor 15,5 %, tristeza 7,9 % | ✗ español |

**1. Negación.** El modelo se fija en palabras clave («sad», «hate») y no capta que «not» o «don't» cambian el sentido: «I am not sad at all» sale tristeza con 96,9 % de confianza. Seguramente en los tuits de entrenamiento casi no había casos así, y por eso no aprendió ese patrón.

**2. Sarcasmo.** «Oh great, another Monday. I'm just thrilled» sale alegría con 93,5 %. Las palabras son positivas, pero el sentido es negativo y depende del contexto («another Monday») y del tono, algo que un modelo entrenado con frases literales no entiende. Lo peor es que se equivoca **con mucha confianza**, así que el umbral no sirve para detectar este error.

**3. Emociones mezcladas.** El modelo solo puede elegir una emoción. En «I'm happy for her but I will miss her terribly» reparte la probabilidad (alegría 60,1 % y tristeza 30,4 %). El gráfico sí muestra la mezcla, pero la etiqueta principal deja de lado la mitad del mensaje.

**4. Clases vecinas.** En la matriz de confusión de prueba, 42 de las 159 frases de **amor** se clasificaron como alegría y 26 de alegría como amor: las dos usan vocabulario positivo parecido. Con **sorpresa** pasa algo parecido: de 66 frases acierta solo 36 (recall 0,545), 14 se van a miedo y 12 a alegría. Por eso «I can't believe what just happened» queda casi empatada entre miedo (41,1 %) y sorpresa (35,9 %), y «I can't believe it, I won the lottery!» sale alegría en vez de sorpresa.

**5. Clases minoritarias.** En el conjunto de prueba hay 695 frases de alegría y 581 de tristeza, pero solo 159 de amor y 66 de sorpresa. En las 4000 frases con las que entrenamos pasa lo mismo: 1341 de alegría y 1160 de tristeza frente a solo 313 de amor y 144 de sorpresa. Con tan pocos ejemplos, amor (F1 0,747) y sorpresa (F1 0,649) son justo las que peor le salen al modelo.

**6. Español.** El modelo y su tokenizador (`uncased`, con vocabulario en inglés) solo vieron inglés. «Tengo mucho miedo de quedarme solo» sale alegría (73,6 %) y «Te odio, me mentiste otra vez» también alegría (67,8 %). «Estoy muy feliz…» sale bien, pero es casualidad: cuando no entiende el texto, el modelo tiende a responder alegría. Por eso la app avisa cuando detecta español.

**7. Errores con mucha confianza.** Entre los 10 errores de prueba con más confianza (todos entre 96 % y 98 %), varios parecen **etiquetas dudosas del propio dataset**: «i feel underappreciated and under valued» está etiquetada como alegría y el modelo dice tristeza (96,6 %); «i feel very saddened that the king…» está etiquetada como alegría y el modelo dice tristeza (97,8 %). Otros son frases largas con varias emociones, donde una palabra fuerte se lleva todo («the depth of sorrow that i feel…», etiquetada sorpresa, predicha tristeza con 96,3 %). O sea, parte de ese 10 % de error no es culpa del modelo sino de etiquetas dudosas.

**Conclusión:** el modelo anda bien con frases directas en inglés, pero se apoya demasiado en palabras clave. Falla con negaciones, sarcasmo, emociones mezcladas, las clases con pocos ejemplos (amor y sorpresa) y cualquier idioma que no sea inglés. Para mejorarlo se podría entrenar con todo el dataset (16 000 frases en vez de 4000), balancear las clases o usar un modelo multilingüe.

## Declaración de uso de IA

- **Herramientas usadas:** Claude (Anthropic), mediante Claude Code.
- **Para qué partes del proyecto:** escribir y corregir `app.py`, `requirements.txt` y `.gitignore`, hacer las pruebas de la app, el análisis de errores con nuestro modelo, las capturas y este README.
- **Qué verificamos o modificamos nosotros:** el modelo lo entrenamos nosotros en el notebook de clase. Revisamos el código línea por línea para poder explicarlo, ejecutamos la app con nuestro modelo y comprobamos que los resultados del análisis de errores coincidieran con lo que muestra la app.
