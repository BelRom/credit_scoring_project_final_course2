from __future__ import annotations

from pathlib import Path
import json

M = Path("reports/metrics_optimization.json")
B = Path("reports/bench_optimization.json")
OUT = Path("reports/REPORT_OPTIMIZATION.md")

def row(name: str, d: dict) -> str:
    return f"| {name} | {d['ms_per_batch']:.3f} | {d['ms_per_sample']:.6f} | {d['samples_per_s']:.1f} |"

def main() -> None:
    metrics = json.loads(M.read_text(encoding="utf-8"))
    bench = json.loads(B.read_text(encoding="utf-8"))

    lines = []
    lines.append("# Model Optimization & Performance Report\n")

    lines.append("## 1. Optimization\n")
    lines.append("- Torch: dynamic INT8 quantization for Linear layers\n")
    lines.append("- ONNX: dynamic INT8 quantization (onnxruntime)\n")

    lines.append("## 2. Quality Metrics (before/after)\n")
    lines.append("| Variant | ROC-AUC | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in metrics.items():
        lines.append(f"| {k} | {v['roc_auc']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |")

    lines.append("\n## 3. Inference Benchmark (CPU)\n")
    lines.append(f"Batch size: {bench['batch_size']}\n")
    lines.append("| Variant | ms/batch | ms/sample | samples/s |")
    lines.append("|---|---:|---:|---:|")
    for k in ["torch_fp32", "torch_quant_dynamic", "onnx_fp32", "onnx_int8"]:
        lines.append(row(k, bench["cpu"][k]))

    lines.append("\n## 4. Inference Benchmark (GPU)\n")
    gpu = bench.get("gpu", {})
    if isinstance(gpu, str):
        lines.append(f"- GPU benchmark not available: {gpu}\n")
    elif "onnx" in gpu and isinstance(gpu["onnx"], str):
        lines.append(f"- ONNX GPU: {gpu['onnx']}\n")
        lines.append("| Variant | ms/batch | ms/sample | samples/s |")
        lines.append("|---|---:|---:|---:|")
        for k in ["torch_fp32", "torch_quant_dynamic"]:
            lines.append(row(k, gpu[k]))
    else:
        lines.append("| Variant | ms/batch | ms/sample | samples/s |")
        lines.append("|---|---:|---:|---:|")
        for k in ["torch_fp32", "torch_quant_dynamic", "onnx_fp32", "onnx_int8"]:
            if k in gpu:
                lines.append(row(k, gpu[k]))

    lines.append("\n## 5. Production Recommendation\n")
    lines.append(
        "- Recommended runtime: ONNXRuntime INT8 on CPU (best price/performance for tabular MLP)\n"
        "- Choose the smallest CPU instance that meets target SLO (p95 latency) under expected RPS\n"
        "- Use GPU only if CPU cannot meet SLO or workload requires very high throughput\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUT}")

if __name__ == "__main__":
    main()
