# ⚡ Optimizing Transformer–Based Time Series Forecasting (TFT)

> **High Performance Machine Learning (HPML) – Spring 2025 Project**  
> **Team:** Aneesh Mokashi (`akm9999`), Ashutosh Agrawal (`aa12398`)

---

## 🧠 Project Summary

### ❓ Problem Statement
The **Temporal Fusion Transformer (TFT)**, while accurate, suffers from **high latency and memory consumption**, limiting its use in real-time or edge environments.

### 💡 Solution
We optimize TFT through:
- 🔄 **FlashAttention** – faster and more efficient attention computation  
- 📉 **Quantization (PTQ)** – model size compression with minimal accuracy drop  
- 🎛️ **Hyperparameter Tuning** – using Optuna to find optimal batch size and attention heads

### 🎯 Value
- ⚡ **Faster Inference**
- 💾 **Lower Memory Footprint**
- 🚀 **Better Deployability**

---

## 🛠️ Tech Stack

- **Frameworks**: PyTorch, GluonTS  
- **Tools**: FlashAttention, Optuna, PyTorch Quantization  
- **Hardware**: NVIDIA A100 / V100 (NYU HPC)

---

## 🧪 Experimental Pipeline

1. **Baseline Setup**: Original TFT with default config  
2. **FlashAttention Integration**: Custom kernel injection for speedup  
3. **Hyperparameter Tuning**: Grid search over batch size & heads  
4. **Quantization (PTQ)**: Compress model while retaining accuracy

---

## 📈 Key Results

- ⏱️ **Training Time Reduction**:  
  - Default: `~3%`  
  - Optimal (128B–4H): `~10.9%`  

- 🧠 **Memory Savings**: FlashAttention reduced peak memory at larger batch sizes  

- 🗜️ **Model Size Drop**:  
  - PTQ reduced size by **up to 74.7%**

- 🚦 **Inference Time**:  
  - Large model: `1660.6s → 955.9s` (**↓ 42.4%**)  
  - Small model: `95.1s → 115.3s` (**↑ 17%** due to dequant overhead)

---

## 📊 Observations & Conclusions

### 🔍 Observations
- FlashAttention benefits scale with batch size
- 128 batch & 4 heads yielded optimal accuracy/performance
- Quantization slightly increased RMSE/MASE but improved speed

### ✅ Conclusions
- FlashAttention significantly improves training/inference in GPU-limited settings
- Quantized models are viable for deployment with negligible accuracy trade-off
- Post-processing & sampling, not attention, are key runtime bottlenecks

---
