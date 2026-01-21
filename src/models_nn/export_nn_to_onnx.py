from __future__ import annotations

from pathlib import Path
import torch

from src.models_nn.nn_runtime import (
    ONNX_PATH,
    TabularMLP,
    load_nn_checkpoint,
)

def main() -> None:
    # Load checkpoint with trained weights
    ckpt = load_nn_checkpoint()

    # Reconstruct model with exact same architecture
    model = TabularMLP(
        in_features=ckpt.in_features,
        hidden_sizes=ckpt.hidden_sizes,
        dropout=ckpt.dropout,
    )
    model.load_state_dict(ckpt.state_dict)
    model.eval()

    # Dummy input (batch=4, features=in_features)
    dummy = torch.zeros(
        (4, ckpt.in_features),
        dtype=torch.float32,
    )

    # Экспорт в ONNX
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH.as_posix(),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["float_input"],
        output_names=["logits"],
        dynamic_axes={
            "float_input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"✅ Exported ONNX to: {ONNX_PATH.as_posix()}")

if __name__ == "__main__":
    main()
