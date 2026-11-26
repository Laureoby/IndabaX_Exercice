import psutil
import os
import time
import nemo.collections.asr as nemo_asr
import soundfile as sf

process = psutil.Process(os.getpid())

def get_memory_mb():
    return process.memory_info().rss / (1024**2)

print("Mémoire avant chargement :", f"{get_memory_mb():.1f} MB")
print("Chargement du modèle...")
start_load = time.time()
model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")
load_time = time.time() - start_load
print(f"Temps de chargement : {load_time:.2f}s")
print("Mémoire après chargement :", f"{get_memory_mb():.1f} MB")

# Mesurer l'inférence sur un fichier
audio_path = "data/audio/afrivoice_00.wav"
info = sf.info(audio_path)

start_infer = time.time()
results = model.transcribe([audio_path])
infer_time = time.time() - start_infer

peak_memory = get_memory_mb()

print(f"\n--- Résultats ---")
print(f"Durée audio : {info.duration:.2f}s")
print(f"Temps d'inférence : {infer_time:.2f}s")
print(f"RTF : {infer_time/info.duration:.4f}")
print(f"Mémoire pic (RSS) : {peak_memory:.1f} MB")
print(f"Transcription : {results[0].text[:100] if hasattr(results[0], 'text') else results[0]}")