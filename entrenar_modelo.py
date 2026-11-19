"""Reproduce el Bloque 4 del notebook de clase y guarda modelo_emociones_distilbert/.

Úsalo solo si no tienes la carpeta del modelo (en Colab con GPU T4 tarda ~2 min).
Equivale a ejecutar el notebook hasta 4.4: mismos hiperparámetros y semilla.
    pip install "transformers>=5.0" datasets accelerate scikit-learn torch
    python entrenar_modelo.py
"""
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

SEMILLA, N_TRAIN, MAX_LEN = 42, 4000, 64
MODELO_BASE = "distilbert-base-uncased"
RUTA_MODELO = "modelo_emociones_distilbert"
ETIQUETAS_ES = ["tristeza", "alegría", "amor", "ira", "miedo", "sorpresa"]
id2label = dict(enumerate(ETIQUETAS_ES))
label2id = {e: i for i, e in id2label.items()}
GPU = torch.cuda.is_available()

set_seed(SEMILLA)
dataset = load_dataset("dair-ai/emotion")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
tokenizar = lambda lote: tokenizer(lote["text"], truncation=True, max_length=MAX_LEN)

train_tok = dataset["train"].shuffle(seed=SEMILLA).select(range(N_TRAIN)).map(tokenizar, batched=True)
val_tok = dataset["validation"].map(tokenizar, batched=True)
test_tok = dataset["test"].map(tokenizar, batched=True)

modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE, num_labels=6, id2label=id2label, label2id=label2id)


def metricas(ep):
    p = np.argmax(ep[0], axis=-1)
    return {"accuracy": accuracy_score(ep[1], p), "f1_macro": f1_score(ep[1], p, average="macro")}


args = TrainingArguments(
    output_dir="salida_emociones", learning_rate=2e-5, per_device_train_batch_size=16,
    per_device_eval_batch_size=64, num_train_epochs=3, weight_decay=0.01, warmup_steps=0.1,
    eval_strategy="epoch", save_strategy="no", logging_steps=25, fp16=GPU,
    report_to="none", seed=SEMILLA)
trainer = Trainer(model=modelo, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                  data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer,
                  compute_metrics=metricas)
trainer.train()
print("Prueba:", trainer.evaluate(test_tok, metric_key_prefix="test"))
trainer.save_model(RUTA_MODELO)   # pesos + configuración + tokenizador
print("Modelo guardado en", RUTA_MODELO)
