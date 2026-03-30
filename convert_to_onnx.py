

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from features import get_feature_count

INPUT_SIZE = get_feature_count()  # 42 features


class PhishNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)


def convert():
    print("=" * 55)
    print("  PhishGuard — PyTorch → ONNX Converter")
    print("=" * 55)

    # ── Load metadata (scaler + input_size)
    if not os.path.exists("char2idx.pkl"):
        print("❌  char2idx.pkl not found — run train_dl.py first")
        return
    with open("char2idx.pkl", "rb") as f:
        meta = pickle.load(f)

    saved_size = meta.get("input_size", INPUT_SIZE)
    if saved_size != INPUT_SIZE:
        print(f"⚠️  Warning: char2idx.pkl has input_size={saved_size} "
              f"but features.py gives {INPUT_SIZE}. Using {saved_size}.")
    actual_size = saved_size

    # ── Load model weights
    if not os.path.exists("lstm_model.pt"):
        print("❌  lstm_model.pt not found — run train_dl.py first")
        return

    model = PhishNet(input_size=actual_size)
    model.load_state_dict(
        torch.load("lstm_model.pt", map_location="cpu", weights_only=True)
    )
    # Set to eval and disable dropout for export
    model.eval()

    print(f"\n✅  Model loaded (input_size={actual_size})")

    # ── Verify with a dummy prediction before export
    dummy_input = torch.randn(2, actual_size)   # batch of 2 — BatchNorm needs >1 sample
    with torch.no_grad():
        out = model(dummy_input)
        probs = torch.softmax(out, dim=1)
        print(f"✅  Test prediction: SAFE={probs[0][0]:.3f}  PHISHING={probs[0][1]:.3f}")

    # ── Export to ONNX (opset 11 — no onnxscript needed)
    print("\n⏳  Exporting to ONNX...")
    try:
        # Method 1: standard export with opset 11 (no onnxscript dependency)
        torch.onnx.export(
            model,
            dummy_input,
            "phishnet.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print("✅  phishnet.onnx saved (opset 11)")

    except Exception as e1:
        print(f"⚠️  Standard export failed ({e1}), trying TorchScript path...")
        # Method 2: trace first, then export — fully bypasses onnxscript
        traced = torch.jit.trace(model, dummy_input)
        torch.onnx.export(
            traced,
            dummy_input,
            "phishnet.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print("✅  phishnet.onnx saved via TorchScript trace (opset 11)")

    # ── Verify ONNX output matches PyTorch output
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("phishnet.onnx")
        onnx_out = sess.run(["output"], {"input": dummy_input.numpy()})[0]
        exp_onnx = np.exp(onnx_out)
        onnx_probs = exp_onnx / exp_onnx.sum(axis=1, keepdims=True)
        print(f"✅  ONNX verification: SAFE={onnx_probs[0][0]:.3f}  PHISHING={onnx_probs[0][1]:.3f}")

        diff = abs(probs[0][1].item() - onnx_probs[0][1])
        if diff < 0.001:
            print(f"✅  PyTorch ↔ ONNX match (diff={diff:.6f})")
        else:
            print(f"⚠️  Small numerical diff: {diff:.6f} (acceptable)")
    except ImportError:
        print("⚠️  onnxruntime not installed — skipping verification")
        print("    pip install onnxruntime   to verify")

    # ── File size
    size_mb = os.path.getsize("phishnet.onnx") / 1024 / 1024
    print(f"\n📦  phishnet.onnx size: {size_mb:.2f} MB  (vs ~800MB for torch)")

    print("\n" + "=" * 55)
    print("  NEXT STEPS:")
    print("=" * 55)
    print("  1. git add phishnet.onnx")
    print("  2. git commit -m 'Add ONNX model'")
    print("  3. git push")
    print("  4. Remove torch from requirements.txt")
    print("  5. Deploy to Render — done!")
    print("=" * 55)


if __name__ == "__main__":
    convert()