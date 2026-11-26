import nemo.collections.asr as nemo_asr
import time
import soundfile as sf

class ASREngine:
    def __init__(self, model_path="DigitalUmuganda/Mbaza-ASR-Afrivoice-660h"):
        # Chargement (mesurer le temps ici)
        start = time.time()
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_path)
        self.load_time = time.time() - start

    def transcribe(self, audio_path):
        # NeMo attend du 16kHz mono
        start = time.time()
        # La méthode transcribe retourne une liste de chaînes
        output = self.model.transcribe([audio_path])
        inference_time = time.time() - start
        
        # Calcul du RTF
        audio_info = sf.info(audio_path)
        duration = audio_info.duration
        rtf = inference_time / duration
        
        return output[0], inference_time, rtf