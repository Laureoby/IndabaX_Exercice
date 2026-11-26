import json
import time
from llama_cpp import Llama

with open("data/llm_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

print("Chargement du modèle LLM...")
llm = Llama(
    #model_path="models/gemma-4-12b-it-qat-Q4_0.gguf",
    model_path="models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0,
    chat_format="gemma",   # ← Ajouté ici, laisse llama.cpp gérer le template Gemma
    verbose=False
)
print("Modèle chargé.")

results = []

for item in prompts:
    start = time.time()
    # ← Remplacé ici : utilise create_chat_completion au lieu d'un prompt brut
    """ output = llm.create_chat_completion(
        messages=[{"role": "user", "content": item['prompt']}],
        max_tokens=200,
        temperature=0.3,
        repeat_penalty=1.3
    ) """

    output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Always respond in Kinyarwanda. Do not use English."},
        {"role": "user", "content": item['prompt']}
    ],
    max_tokens=200,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    repeat_penalty=1.08
    )
    
    elapsed = time.time() - start

    response_text = output["choices"][0]["message"]["content"].strip()
    tokens_generated = output["usage"]["completion_tokens"]

    results.append({
        "id": item["id"],
        "task": item["task"],
        "prompt": item["prompt"],
        "expected_behavior": item["expected_behavior"],
        "response": response_text,
        "generation_time_s": round(elapsed, 2),
        "tokens_generated": tokens_generated,
        "tokens_per_second": round(tokens_generated / elapsed, 2) if elapsed > 0 else 0
    })

    print(f"\n--- Prompt {item['id']} ({item['task']}) ---")
    print(f"Réponse : {response_text[:300]}")
    print(f"Temps : {elapsed:.2f}s | Tokens : {tokens_generated}")

with open("results/llm_baseline.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nRésultats sauvegardés dans results/llm_baseline.json")