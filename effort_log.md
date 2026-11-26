# Effort Log

Approximate time spent, obstacles, assistance used, and unfinished work. Times are rough estimates.

## Timeline

| Phase | Approx. time | Notes |
|-------|--------------|-------|
| Environment setup (venv, NeMo, PyTorch) | 2 h | NeMo install is heavy; several dependency conflicts (numpy, datasets). |
| Finding Kinyarwanda evaluation audio | 3 h | Common Voice HF repos were emptied; several mirrors were broken; finally used `Kira-Floris/Afrivoice-Kinyarwanda-ASR` (Parquet mirror of `DigitalUmuganda/Afrivoice_Kinyarwanda`). |
| Extracting 10 clips + references | 1.5 h | WebM/Opus audio had to be decoded with ffmpeg; `soundfile` could not read it directly. |
| ASR baseline evaluation | 1 h | WER 0.2021, RTF 0.0721. |
| LLM attempts | 3 h | Transformers path exceeded memory; llama.cpp + community GGUF produced degenerate outputs; documented in report. |
| Gradio interface | 1 h | Works offline for ASR; LLM step is a placeholder. |
| ONNX optimization | 1 h | [Outcome: success / failed — describe]. |
| Documentation (README, report, effort log) | 1.5 h | |
| Total | ≈14 h | Within the 16 h limit. |

## Obstacles encountered

1. Common Voice HF repositories emptied: `mozilla-foundation/common_voice_17_0` returns `EmptyDatasetError`; `fsicoli/common_voice_22_0` uses a loading script that `datasets>=4` no longer supports.
2. `datasets` version conflict: NeMo requires `datasets>=4`, but the Afrivoice loading script requires `datasets<4`. Resolved by using a Parquet mirror instead of the script.
3. WebM/Opus audio: `soundfile` cannot decode it; used `ffmpeg` via `subprocess`.
4. Gemma-4 GGUF defects: community GGUF files produce degenerate outputs (repetition, leaked thinking tags), consistent with reported tensor-permutation issues. The 23 GB unquantized checkpoint does not fit in this laptop's memory.

## Assistance used

- Hugging Face documentation, NeMo documentation, `jiwer` documentation.
- Public discussions of Gemma-4 GGUF quantization issues (to diagnose the degenerate outputs).
- AI assistant for code scaffolding and debugging; all code was run and verified locally.

## Unfinished work

- LLM quality scoring (blocked by memory and GGUF defects).
- ONNX INT8 WER re-measurement (depends on whether ONNX inference succeeds).
- Mobile deployment checks (out of scope, but discussed in the report).

## Notes

- No audio was uploaded to external services.
- No credentials are included in the repository.


## Obstacles encountered

4. Transformers memory error: the 23.9 GB unquantized checkpoint could not be mmap'd (`Cannot allocate memory`). Resolved by using a GGUF Q4_K_XL with llama.cpp, which loads progressively.

## Unfinished work
- ONNX INT8 WER re-measurement (depends on whether ONNX export succeeds).