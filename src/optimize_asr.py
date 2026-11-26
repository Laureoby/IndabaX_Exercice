import nemo.collections.asr as nemo_asr
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
print(f"Réduction : {100 * (1 - size_optimized/size_baseline):.1f}%")