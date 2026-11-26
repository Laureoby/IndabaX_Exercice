import json
import time
from llama_cpp import Llama

# 1. Charger les prompts
with open("data/llm_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

MODEL_PATH = "models/gemma-4-12b-it-UD-Q4_K_XL.gguf"

print("Chargement du modèle GGUF...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,          # Contexte réduit pour économiser la RAM
    n_threads=4,         # Ajustez selon votre CPU (nproc)
    n_gpu_layers=0,      # CPU uniquement (pas de GPU détecté)
    verbose=False
)
print("Modèle chargé.")

results = []

for item in prompts:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in Kinyarwanda."},
        {"role": "user", "content": item['prompt']}
    ]

    start = time.time()
    try:
        output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )
        
        elapsed = time.time() - start

        response_text = output["choices"][0]["message"]["content"].strip()

        results.append({
            "id": item["id"],
            "task": item["task"],
            "prompt": item["prompt"],
            "expected_behavior": item["expected_behavior"],
            "response": response_text,
            "generation_time_s": round(elapsed, 2)
        })

        print(f"\n--- Prompt {item['id']} ---")
        print(f"Réponse : {response_text[:300]}")
        print(f"Temps : {elapsed:.2f}s")

    except Exception as e:
        print(f"\n--- Prompt {item['id']} : ERREUR ---")
        print(f"{type(e).__name__}: {e}")
        results.append({
            "id": item["id"],
            "task": item["task"],
            "prompt": item["prompt"],
            "expected_behavior": item["expected_behavior"],
            "response": None,
            "error": f"{type(e).__name__}: {e}",
            "generation_time_s": None
        })

with open("results/llm_baseline_transformers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nRésultats sauvegardés dans results/llm_baseline_transformers.json")