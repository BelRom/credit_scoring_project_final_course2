from __future__ import annotations

from pathlib import Path
import tempfile

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

IN_ONNX = Path("models/nn_model.onnx")
OUT_ONNX = Path("models/nn_model_int8.onnx")


def main() -> None:
    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(IN_ONNX.as_posix())

    # protobuf container: clear через del
    del model.graph.value_info[:]

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "clean.onnx"
        onnx.save(model, tmp_path.as_posix())

        quantize_dynamic(
            model_input=tmp_path.as_posix(),
            model_output=OUT_ONNX.as_posix(),
            weight_type=QuantType.QInt8,
            extra_options={"DisableShapeInference": True},
        )

    print(f"Saved: {OUT_ONNX}")


if __name__ == "__main__":
    main()
