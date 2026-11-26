""" import nemo.collections.asr as nemo_asr
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Charger le modèle baseline
print("Chargement du modèle ASR...")
model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")
model.eval()

# 2. Exporter en ONNX (chemin de sortie)
onnx_path = "models/asr_baseline.onnx"
print(f"Export ONNX vers {onnx_path}...")
model.export(onnx_path)
print("Export terminé.")

# 3. Quantifier en INT8 (poids)
quantized_path = "models/asr_optimized_int8.onnx"
print(f"Quantification INT8 vers {quantized_path}...")
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_path,
    weight_type=QuantType.QInt8
)
print("Quantification terminée.")

# 4. Comparer les tailles
size_baseline = os.path.getsize(onnx_path) / (1024**2)
size_optimized = os.path.getsize(quantized_path) / (1024**2)
print(f"\nTaille baseline : {size_baseline:.2f} MB")
print(f"Taille optimisée : {size_optimized:.2f} MB")
print(f"Réduction : {100 * (1 - size_optimized/size_baseline):.1f}%") """

import nemo.collections.asr as nemo_asr
import os
import traceback
from onnxruntime.quantization import quantize_dynamic, QuantType

# Chemins de sortie
ONNX_PATH = "models/asr_baseline.onnx"
QUANTIZED_PATH = "models/asr_optimized_int8.onnx"

# 1. Charger le modèle baseline
print("Chargement du modèle ASR...")
model = nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")
model.eval()
model.to("cpu")  # Forcer le CPU pour éviter les problèmes CUDA/ONNX
print("Modèle chargé.")

# 2. Exporter en ONNX avec des paramètres conservateurs
print(f"Export ONNX vers {ONNX_PATH}...")
export_success = False
try:
    model.export(
        ONNX_PATH,
        onnx_opset_version=17,   # Version d'opset stable
        check_trace=False,       # Désactiver la vérification qui peut planter
        verbose=False
    )
    export_success = True
    print("Export ONNX terminé.")
except Exception as e:
    print(f"Échec de l'export ONNX : {type(e).__name__}: {e}")
    traceback.print_exc()

# 3. Quantifier si l'export a réussi
if export_success and os.path.exists(ONNX_PATH):
    print(f"Quantification INT8 vers {QUANTIZED_PATH}...")
    try:
        quantize_dynamic(
            model_input=ONNX_PATH,
            model_output=QUANTIZED_PATH,
            weight_type=QuantType.QInt8
        )
        print("Quantification terminée.")
    except Exception as e:
        print(f"Échec de la quantification : {type(e).__name__}: {e}")
        traceback.print_exc()

    # 4. Comparer les tailles
    if os.path.exists(QUANTIZED_PATH):
        size_baseline = os.path.getsize(ONNX_PATH) / (1024**2)
        size_optimized = os.path.getsize(QUANTIZED_PATH) / (1024**2)
        print(f"\n--- Résultats ---")
        print(f"Taille baseline ONNX : {size_baseline:.2f} MB")
        print(f"Taille optimisée INT8 : {size_optimized:.2f} MB")
        print(f"Réduction : {100 * (1 - size_optimized/size_baseline):.1f}%")
    else:
        print("\nLe fichier quantifié n'a pas été produit.")
else:
    print("\nL'export ONNX a échoué. La quantification n'a pas été tentée.")
    print("Consultez le message d'erreur ci-dessus et documentez-le dans effort_log.md et report.md.")