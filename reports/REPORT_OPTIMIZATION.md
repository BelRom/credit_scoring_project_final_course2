# Model Optimization & Performance Report

## 1. Optimization

- Torch: dynamic INT8 quantization for Linear layers

- ONNX: dynamic INT8 quantization (onnxruntime)

## 2. Quality Metrics (before/after)

| Variant | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| torch_fp32 | 0.7970 | 0.6616 | 0.3934 | 0.4934 |
| torch_quant_dynamic | 0.7844 | 0.6741 | 0.3647 | 0.4733 |
| onnx_fp32 | 0.7970 | 0.6616 | 0.3934 | 0.4934 |
| onnx_int8 | 0.7960 | 0.6641 | 0.3964 | 0.4965 |

## 3. Inference Benchmark (CPU)

Batch size: 1024

| Variant | ms/batch | ms/sample | samples/s |
|---|---:|---:|---:|
| torch_fp32 | 0.388 | 0.000379 | 2637595.8 |
| torch_quant_dynamic | 0.369 | 0.000360 | 2777967.0 |
| onnx_fp32 | 0.473 | 0.000462 | 2162977.5 |
| onnx_int8 | 0.233 | 0.000228 | 4391363.8 |

## 4. Inference Benchmark (GPU)

| Variant | ms/batch | ms/sample | samples/s |
|---|---:|---:|---:|
| torch_fp32 | 0.071 | 0.000069 | 14467411.7 |
| onnx_fp32 | 0.077 | 0.000075 | 13376159.7 |
| onnx_int8 | 0.509 | 0.000497 | 2011347.8 |

## 5. Production Recommendation

- Recommended runtime: ONNXRuntime INT8 on CPU (best price/performance for tabular MLP)
- Choose the smallest CPU instance that meets target SLO (p95 latency) under expected RPS
- Use GPU only if CPU cannot meet SLO or workload requires very high throughput

