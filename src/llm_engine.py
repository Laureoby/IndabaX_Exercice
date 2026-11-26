# src/llm_engine.py (extrait)
from llama_cpp import Llama
import time

class LLMEngine:
    def __init__(self, model_path):
        # model_path pointe vers le .gguf téléchargé
        start = time.time()
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
        self.load_time = time.time() - start

    def generate(self, prompt, max_tokens=256):
        start = time.time()
        output = self.llm(prompt, max_tokens=max_tokens, stop=["<turn|>"], echo=False)
        elapsed = time.time() - start
        
        text = output['choices'][0]['text']
        tokens = output['usage']['completion_tokens']
        return text, elapsed, tokens