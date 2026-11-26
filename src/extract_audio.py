from datasets import load_dataset
import subprocess
import numpy as np
import soundfile as sf
import os
import json

# 1. Charger le dataset en streaming
ds = load_dataset(
    "Kira-Floris/Afrivoice-Kinyarwanda-ASR",
    name="default",
    split="health_train",
    streaming=True
)

# 2. Créer le dossier de sortie
os.makedirs("data/audio", exist_ok=True)

# 3. Préparer la liste des transcriptions de référence
references = []
count = 0

# 4. Boucler sur les échantillons
for sample in ds:
    if count >= 10:
        break

    audio_info = sample["audio"]
    transcription = sample.get("transcription") or sample.get("text") or sample.get("raw_text")

    if not transcription:
        continue

    # Extraire les octets audio
    if "bytes" not in audio_info or audio_info["bytes"] is None:
        print(f"Échantillon {count} : pas de bytes, ignoré")
        continue

    audio_bytes = audio_info["bytes"]

    # Décoder avec ffmpeg depuis stdin vers stdout, en PCM 16-bit 16kHz mono
    process = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "pipe:1"
        ],
        input=audio_bytes,
        capture_output=True
    )

    if process.returncode != 0:
        print(f"Échec du décodage ffmpeg pour l'échantillon {count} : {process.stderr.decode()[:200]}")
        continue

    # Convertir les octets PCM en tableau numpy float32
    audio_array = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    # Écrire le fichier WAV
    wav_path = f"data/audio/afrivoice_{count:02d}.wav"
    sf.write(wav_path, audio_array, 16000)

    # Enregistrer la transcription de référence
    references.append({
        "audio_file": wav_path,
        "transcription": transcription
    })

    print(f"[{count}] {wav_path} | {transcription[:60]}...")
    count += 1

# 5. Sauvegarder les transcriptions de référence dans un JSON
with open("data/asr_references.json", "w", encoding="utf-8") as f:
    json.dump(references, f, ensure_ascii=False, indent=2)

print(f"\nTerminé : {count} fichiers audio écrits dans data/audio/")
print(f"Transcriptions de référence sauvegardées dans data/asr_references.json")