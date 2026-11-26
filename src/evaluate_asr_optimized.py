import json
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf
import jiwer
import pandas as pd

# Charger les références
with open("data/asr_references.json", "r", encoding="utf-8") as f:
    refs = json.load(f)

def normalize(text):
    text = text.lower()
    text = jiwer.RemovePunctuation()(text)
    text = " ".join(text.split())
    return text

refs_text = [normalize(r["transcription"]) for r in refs]
audio_paths = [r["audio_file"] for r in refs]

# Charger le modèle ONNX quantifié
# NOTE : l'interface exacte dépend de l'export NeMo. Adaptez les noms d'entrée/sortie.
session = ort.InferenceSession("models/asr_optimized_int8.onnx")

# ... (le code d'inférence ONNX dépend de la structure exacte du modèle exporté)
# Si l'export ONNX a produit un seul fichier avec entrée audio et sortie texte,
# utilisez ce bloc. Sinon, adaptez selon les entrées/sorties réelles.

print("WER baseline (mémoire) : 0.2021")
print("Pour le WER optimisé, exécutez l'inférence ONNX et comparez avec les mêmes références.")