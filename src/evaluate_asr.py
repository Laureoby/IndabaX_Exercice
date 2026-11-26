import nemo.collections.asr as nemo_asr
import json
import jiwer
import time
import soundfile as sf
import pandas as pd

# Charger les références
with open("data/asr_references.json", "r", encoding="utf-8") as f:
    refs = json.load(f)

# Charger le modèle
model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")

# Normalisation cohérente pour références et prédictions
def normalize(text):
    text = text.lower()
    text = jiwer.RemovePunctuation()(text)
    text = " ".join(text.split())
    return text

refs_text = [normalize(r["transcription"]) for r in refs]
audio_paths = [r["audio_file"] for r in refs]

# Transcrire tous les fichiers
results = model.transcribe(audio_paths, batch_size=1)
preds_text = [normalize(r.text) for r in results]

# Calculer le WER global
wer = jiwer.wer(refs_text, preds_text)

# Calculer le RTF (durée totale d'inférence / durée totale audio)
total_audio_duration = 0
for path in audio_paths:
    info = sf.info(path)
    total_audio_duration += info.duration

# Mesurer le temps d'inférence total (hors chargement)
start = time.time()
_ = model.transcribe(audio_paths, batch_size=1)
inference_time = time.time() - start
rtf = inference_time / total_audio_duration

# Enregistrer les résultats détaillés
rows = []
for ref, pred, path in zip(refs_text, preds_text, audio_paths):
    rows.append({
        "audio_file": path,
        "reference": ref,
        "prediction": pred,
        "wer": jiwer.wer(ref, pred)
    })

df = pd.DataFrame(rows)
df.to_csv("results/asr_baseline.csv", index=False)

print(f"WER global : {wer:.4f}")
print(f"RTF : {rtf:.4f}")
print(f"Durée audio totale : {total_audio_duration:.2f} s")
print(f"Temps d'inférence total : {inference_time:.2f} s")
print(f"Résultats détaillés sauvegardés dans results/asr_baseline.csv")