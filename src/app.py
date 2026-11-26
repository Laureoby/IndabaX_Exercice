import gradio as gr
import nemo.collections.asr as nemo_asr
import soundfile as sf
import os
import time

# Chargement du modèle ASR au démarrage (mesurer le temps)
print("Chargement du modèle ASR...")
start = time.time()
asr_model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")
load_time = time.time() - start
print(f"Modèle chargé en {load_time:.2f}s")

def process_audio(audio_path):
    if audio_path is None:
        return "Erreur : Aucun fichier audio fourni.", ""
    
    try:
        # Vérifier que le fichier existe et n'est pas vide
        if not os.path.exists(audio_path):
            return "Erreur : Fichier introuvable.", ""
        
        info = sf.info(audio_path)
        if info.duration < 0.5:
            return "Erreur : Audio trop court (moins de 0,5s).", ""
        
        # Transcrire
        start_infer = time.time()
        results = asr_model.transcribe([audio_path])
        infer_time = time.time() - start_infer
        
        transcription = results[0].text if hasattr(results[0], 'text') else str(results[0])
        
        # Placeholder pour le LLM (le GGUF actuel est défectueux)
        llm_response = f"[LLM non disponible dans cette configuration — GGUF défectueux]\n\nTranscription : {transcription}"
        
        return transcription, llm_response
    
    except Exception as e:
        return f"Erreur : {str(e)}", ""

# Interface Gradio
with gr.Blocks(title="Pipeline Vocal Hors-ligne (Kinyarwanda)") as demo:
    gr.Markdown("# Pipeline Vocal Hors-ligne: ASR Kinyarwanda")
    gr.Markdown("Enregistrez ou téléversez un fichier audio (.wav, 16kHz mono).")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
    
    with gr.Row():
        btn = gr.Button("Traiter", variant="primary")
        clear_btn = gr.Button("Effacer")
    
    with gr.Row():
        transcription_out = gr.Textbox(label="Transcription ASR", lines=3)
        response_out = gr.Textbox(label="Réponse (LLM placeholder)", lines=5)
    
    btn.click(fn=process_audio, inputs=audio_input, outputs=[transcription_out, response_out])
    clear_btn.click(fn=lambda: (None, "", ""), outputs=[audio_input, transcription_out, response_out])

demo.launch(server_name="0.0.0.0", share=False)