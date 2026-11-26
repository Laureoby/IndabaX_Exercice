import nemo.collections.asr as nemo_asr
import time

print("Chargement du modèle ASR...")
start = time.time()
model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")
load_time = time.time() - start
print(f"Chargement terminé, durée : {load_time:.2f} s")

# Transcrire un fichier audio
wav_path = "data/audio/afrivoice_00.wav"
start = time.time()
results = model.transcribe([wav_path])
infer_time = time.time() - start

print(f"Transcription : {results[0]}")
print(f"Durée d'inférence : {infer_time:.2f} s")