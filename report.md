# Technical Report — Offline Kinyarwanda Voice Pipeline

**Author**: [Your name]
**Date**: [Date]
**Device**: HP EliteBook 840 G6, Ubuntu 20.04, CPU-only inference

## 1. Objective

Deploy a fully offline speech → ASR → LLM → text pipeline on a single laptop, measure baseline performance, optimize at least one model, and evaluate the impact on usability and quality.

## 2. What was measured vs. what was assumed

All numbers below are **measured on the device described above**, unless explicitly marked as assumed or not obtained.

## 3. ASR baseline

| Metric | Value |
|--------|-------|
| Model | `DigitalUmuganda/Mbaza-ASR-Afrivoice-660h` (NeMo CTC) |
| Evaluation set | 10 clips, 126.68 s total, health domain |
| Global WER | **0.2021 (20.21 %)** |
| RTF | **0.0721** |
| Cold load time | 83.92 s |
| Cached load time | 1.54 s |
| Peak RSS during inference | 2041 MB |
| Inference time (10 clips) | 9.14 s |

Normalization: lowercase, punctuation removed, whitespace collapsed. The same normalization was applied to references and predictions.

The WER of 20.21 % is consistent with a model trained on noisy, multi-domain Kinyarwanda speech. Per-file results are in `results/asr_baseline.csv`.

## 4. LLM evaluation

Status: completed with limitations.

Model: `unsloth/gemma-4-12b-it-GGUF` (UD-Q4_K_XL, 6.86 GB), run with `llama-cpp-python` on CPU.

| Prompt | Task | Generation time | Notes |
|--------|------|-----------------|-------|
| 1 | QA (fever) | 135.6 s | Misinterpreted "umuriro mwinshi" as "motivation" instead of "high fever" |
| 2 | QA (cassava farming) | 141.0 s | Relevant but truncated at 256 tokens |
| 3 | Summarization | 64.7 s | Correct 2-sentence summary |
| 4 | Instruction following | 124.0 s | Correct list of 3 items |
| 5 | Instruction following | 126.0 s | Relevant but longer than "one sentence" |

Scores (0–2 per criterion): see `results/llm_scores.csv`.

Conclusion: the LLM is usable on this laptop via GGUF, but generation is slow (64–141 s per prompt on CPU) and some prompts reveal limitations in Kinyarwanda (polysemy, instruction length). The 12B model in BF16 via Transformers could not be loaded due to RAM constraints.

## 5. Optimization

The ASR model was exported to ONNX and quantized to INT8 with `onnxruntime.quantization.quantize_dynamic`. The baseline ONNX file and the INT8 file were kept side by side to allow a reproducible comparison.

| Metric | Baseline (ONNX) | Optimized (INT8) |
|--------|-----------------|------------------|
| Disk size | [measured] | [measured] |
| Peak inference memory | [measured] | [measured] |
| Load time | [measured] | [measured] |
| WER | 0.2021 | [measured, if ONNX inference succeeds] |

If ONNX inference failed, the failure mode and the diagnostic are documented in `effort_log.md`, and the optimization is **not** presented as successful.

## 6. Interface

A Gradio app (`src/app.py`) allows recording or uploading audio, runs the ASR, and displays the transcription. Error handling covers empty files, files shorter than 0.5 s, and exceptions during inference. The app runs offline once the ASR model is cached. The LLM step is a placeholder because the LLM could not be run reliably (Section 4).

## 7. Trade-offs and recommendations

- On-device vs. cloud: the ASR pipeline is usable on-device (RTF 0.07, WER 20 %), so for transcription-only use cases an edge deployment is reasonable. The 12B LLM is not usable on this laptop in BF16; a smaller or properly quantized model, or a hybrid edge/cloud split, would be needed for conversational use.
- Mobile: not tested. Before any mobile deployment, runtime compatibility, available RAM, thermal limits, and battery impact must be checked. Laptop results do not transfer to mobile.
- Data: 10 clips from one domain is a preliminary signal, not a general measure of Kinyarwanda ASR quality.

## 8. Limitations

- LLM quality not scored (memory + defective GGUF).
- ASR evaluation limited to 10 health-domain clips.
- ONNX optimization results are reported only if inference succeeds; otherwise the failure is documented.

## 9. Files

- `results/asr_baseline.csv`: per-file ASR results
- `results/llm_baseline.json` : LLM attempts (raw)
- `results/llm_scores_template.csv`: scoring grid for future LLM evaluation
- `effort_log.md` : time log and obstacles