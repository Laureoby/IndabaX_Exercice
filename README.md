# Offline Kinyarwanda Voice Pipeline: ASR + LLM on a Single Laptop

This repository contains the solution for the Digital Umuganda technical challenge: deploy and optimize an offline voice AI application (Speech → Local ASR → Local LLM → Text response) on a single laptop.

## 1. Project overview

The pipeline is:

1. ASR: `DigitalUmuganda/Mbaza-ASR-Afrivoice-660h` (NeMo CTC model) transcribes Kinyarwanda speech.
2. LLM: `google/gemma-4-12B-it-qat-q4_0-unquantized` (loaded via Transformers) generates a text response.
3. Interface: a Gradio app allows recording or uploading audio, running the pipeline, and displaying the transcription and response.

Everything runs locally. No cloud inference, no remote API.

## 2. Hardware and environment

- Device: HP EliteBook 840 G6 (CPU-only inference; no dedicated GPU used).
- OS: Ubuntu 20.04.
- Python: 3.12 (virtual environment `.venv`).
- Key libraries: NeMo Toolkit, PyTorch, Transformers, Gradio, jiwer, soundfile.

Exact versions are listed in `requirements.txt`.

## 3. Installation
```bash
# Clone the repository
git clone <https://github.com/Laureoby/IndabaX_Exercice.git>
cd IndabaX_Exercice

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## 4. Downloading the models
# ASR model (cached automatically by NeMo on first use)
python3 -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('DigitalUmuganda/Mbaza-ASR-Afrivoice-660h')"

# LLM model (GGUF, ~6.86 GB)
hf download unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-UD-Q4_K_XL.gguf --local-dir ./models

## 5. Results: LLM evaluation
```bash
python3 src/evaluate_llm.py
Outputs: results/llm_baseline.json