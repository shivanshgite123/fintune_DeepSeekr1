# 🚀 Deepseek LLM Fine-Tuning with LoRA

> **Fine-tune the Deepseek 7B Chat model using LoRA (PEFT) on the Alpaca dataset**

---

## 📋 Quick Links
- [🎯 Overview](#overview)
- [⚙️ Prerequisites](#prerequisites)
- [📦 Installation](#installation)
- [🏃 Quick Start](#quick-start)
- [💾 Load Model](#load-model)
- [🐛 Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This is a complete fine-tuning pipeline for **Deepseek 7B Chat** using **LoRA (Low-Rank Adaptation)** on the **Alpaca dataset**.

### ✨ Features

✅ Memory Efficient — LoRA only trains adapter weights  
✅ Fast Training — FP16 precision + gradient accumulation  
✅ Production Ready — Saves model and tokenizer  
✅ Resume Support — Continue from checkpoints  
✅ Chat Format — Instruction-response pairs  

---

## ⚙️ Prerequisites

### Hardware
- GPU VRAM: 20+ GB (RTX 4090, A100, etc.)
- CPU RAM: 16+ GB
- Disk Space: 50+ GB
- OS: Windows with PowerShell

### Software
- Python 3.8+
- Hugging Face account (optional)

---

## 📦 Installation

### 1. Create Virtual Environment

```powershell
python -m venv deepseek_env
.\deepseek_env\Scripts\Activate.ps1
