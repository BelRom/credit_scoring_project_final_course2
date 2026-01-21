# Model Optimization & Performance Report

## 1. Optimization

- Torch: dynamic INT8 quantization for Linear layers

- ONNX: dynamic INT8 quantization (onnxruntime)

## 2. Quality Metrics (before/after)

| Variant | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| torch_fp32 | 0.7866 | 0.6537 | 0.3911 | 0.4894 |
| torch_quant_dynamic | 0.7782 | 0.6625 | 0.3610 | 0.4673 |
| onnx_fp32 | 0.7866 | 0.6537 | 0.3911 | 0.4894 |
| onnx_int8 | 0.7852 | 0.6510 | 0.3851 | 0.4839 |

## 3. Inference Benchmark (CPU)

Batch size: 1024

| Variant | ms/batch | ms/sample | samples/s |
|---|---:|---:|---:|
| torch_fp32 | 0.366 | 0.000358 | 2796202.5 |
| torch_quant_dynamic | 0.379 | 0.000370 | 2700390.1 |
| onnx_fp32 | 0.259 | 0.000253 | 3948644.3 |
| onnx_int8 | 0.185 | 0.000181 | 5524230.9 |

## 4. Inference Benchmark (GPU)

| Variant | ms/batch | ms/sample | samples/s |
|---|---:|---:|---:|
| torch_fp32 | 0.073 | 0.000071 | 14042711.5 |
| onnx_fp32 | 0.076 | 0.000074 | 13462434.2 |
| onnx_int8 | 0.916 | 0.000895 | 1117740.1 |

## 5. Production Recommendation

- Recommended runtime: ONNXRuntime INT8 on CPU (best price/performance for tabular MLP)
- Choose the smallest CPU instance that meets target SLO (p95 latency) under expected RPS
- Use GPU only if CPU cannot meet SLO or workload requires very high throughput

