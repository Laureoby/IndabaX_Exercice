import json
import time
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# 1. Charger les prompts
with open("data/llm_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

# 2. Charger le modèle officiel (chemin local si déjà téléchargé, sinon utiliser l'ID)
MODEL_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized"

print("Chargement du processeur...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Chargement du modèle (peut prendre plusieurs minutes)...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",          # Laisser transformers choisir automatiquement
    device_map="auto",     # Répartir automatiquement entre CPU/GPU
    attn_implementation="sdpa"  # Accélération d'attention
)
model.eval()
print("Modèle chargé.")

results = []

for item in prompts:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in Kinyarwanda."},
        {"role": "user", "content": item['prompt']}
    ]

    # Formatage officiel, désactiver la réflexion
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False   # Clé : désactiver la réflexion
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True
        )
    elapsed = time.time() - start

    # Analyser la sortie
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    # Utiliser parse_response officiel pour supprimer les balises de réflexion
    parsed = processor.parse_response(response, prefix=inputs["input_ids"])

    results.append({
        "id": item["id"],
        "task": item["task"],
        "prompt": item["prompt"],
        "expected_behavior": item["expected_behavior"],
        "response": parsed,
        "generation_time_s": round(elapsed, 2)
    })

    print(f"\n--- Prompt {item['id']} ---")
    print(f"Réponse : {parsed[:300]}")
    print(f"Temps : {elapsed:.2f}s")

with open("results/llm_baseline_transformers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nRésultats sauvegardés dans results/llm_baseline_transformers.json")