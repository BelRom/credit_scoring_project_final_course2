from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from src.models_nn.nn_runtime import TabularMLP, TORCH_CKPT_PATH

OUT_PATH = Path("models/nn_model_quant_dynamic.pt")


def main() -> None:
    ckpt = torch.load(TORCH_CKPT_PATH.as_posix(), map_location="cpu")

    model = TabularMLP(
        in_features=int(ckpt["in_features"]),
        hidden_sizes=tuple(int(x) for x in ckpt.get("hidden_sizes", (256, 128, 64))),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    qmodel.eval()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": qmodel.state_dict(),
            "in_features": int(ckpt["in_features"]),
            "hidden_sizes": tuple(int(x) for x in ckpt.get("hidden_sizes", (256, 128, 64))),
            "dropout": float(ckpt.get("dropout", 0.2)),
            "quantization": "dynamic_int8_linear",
        },
        OUT_PATH.as_posix(),
    )

    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
