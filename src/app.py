import gradio as gr
import nemo.collections.asr as nemo_asr
from llama_cpp import Llama
import soundfile as sf
import os

# --- Chargement des modèles ---
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

def process_audio(audio_path):
    if audio_path is None:
        return "Erreur : Aucun fichier audio fourni.", ""
    try:
        if not os.path.exists(audio_path):
            return "Erreur : Fichier introuvable.", ""
        info = sf.info(audio_path)
        if info.duration < 0.5:
            return "Erreur : Audio trop court (moins de 0,5s).", ""
        results = asr_model.transcribe([audio_path])
        transcription = results[0].text if hasattr(results[0], 'text') else str(results[0])
        llm_response = generate_response(transcription)
        return transcription, llm_response
    except Exception as e:
        return f"Erreur : {str(e)}", ""

# --- CSS personnalisé ---
custom_css = """
#main-title {
    text-align: center;
    font-size: 2.2em;
    font-weight: bold;
    text-transform: uppercase;
    margin-bottom: 10px;
}
#footer {
    text-align: center;
    font-size: 0.9em;
    color: #666;
    margin-top: 20px;
}
"""

with gr.Blocks(title="Pipeline Vocal Hors-ligne", css=custom_css) as demo:
    # Titre avec logos à gauche et à droite
    with gr.Row():
        gr.Image("assets/digiU.svg", show_label=False, width=80, height=80)
        gr.Markdown("# PIPELINE VOCAL HORS LIGNE: ASR + LLM Kinyarwanda", elem_id="main-title")
        gr.Image("assets/IndabaX.png", show_label=False, width=80, height=80)

    gr.Markdown("Enregistrez ou téléversez un fichier audio (.wav, 16kHz mono).")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")

    with gr.Row():
        btn = gr.Button("Traiter", variant="primary", elem_classes="green-btn")
        clear_btn = gr.Button("Effacer")

    with gr.Row():
        transcription_out = gr.Textbox(label="Transcription ASR", lines=3)
        response_out = gr.Textbox(label="Réponse LLM", lines=5)

    btn.click(fn=process_audio, inputs=audio_input, outputs=[transcription_out, response_out])
    clear_btn.click(fn=lambda: (None, "", ""), outputs=[audio_input, transcription_out, response_out])

demo.launch(server_name="0.0.0.0", share=False)
