import gradio as gr
import nemo.collections.asr as nemo_asr
from llama_cpp import Llama
import soundfile as sf
import os
import time

# --- Chargement des modèles au démarrage ---
print("Chargement du modèle ASR...")
asr_model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")

print("Chargement du LLM GGUF...")
llm = Llama(
    model_path="models/gemma-4-12b-it-UD-Q4_K_XL.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)
print("Modèles chargés.")

# --- Fonction de génération LLM ---
def generate_response(transcription):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in Kinyarwanda."},
        {"role": "user", "content": transcription}
    ]
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )
    return output["choices"][0]["message"]["content"].strip()

# --- Fonction appelée par le bouton "Traiter" ---
def process_audio(audio_path):
    if audio_path is None:
        return "Erreur : Aucun fichier audio fourni.", ""
    
    try:
        if not os.path.exists(audio_path):
            return "Erreur : Fichier introuvable.", ""
        
        info = sf.info(audio_path)
        if info.duration < 0.5:
            return "Erreur : Audio trop court (moins de 0,5s).", ""
        
        # 1. ASR
        results = asr_model.transcribe([audio_path])
        transcription = results[0].text if hasattr(results[0], 'text') else str(results[0])
        
        # 2. LLM ← C'est ici qu'on remplace le placeholder
        llm_response = generate_response(transcription)
        
        return transcription, llm_response
    
    except Exception as e:
        return f"Erreur : {str(e)}", ""

# --- Interface Gradio ---
with gr.Blocks(title="Pipeline Vocal Hors-ligne (Kinyarwanda)") as demo:
    gr.Markdown("# Pipeline Vocal Hors-ligne: ASR + LLM Kinyarwanda")
    gr.Markdown("Enregistrez ou téléversez un fichier audio (.wav, 16kHz mono).")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
    
    with gr.Row():
        btn = gr.Button("Traiter", variant="primary")
        clear_btn = gr.Button("Effacer")
    
    with gr.Row():
        transcription_out = gr.Textbox(label="Transcription ASR", lines=3)
        response_out = gr.Textbox(label="Réponse LLM", lines=5)
    
    btn.click(fn=process_audio, inputs=audio_input, outputs=[transcription_out, response_out])
    clear_btn.click(fn=lambda: (None, "", ""), outputs=[audio_input, transcription_out, response_out])

demo.launch(server_name="0.0.0.0", share=False)