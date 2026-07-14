# 🧠 Zorther LLM

> **My latest Large Language Model (LLM) project — Zorther.**
>
> Zorther is a decoder-only Transformer architecture built from scratch with a modular design, optimized for efficient training and inference on modern CPUs, especially AMD Ryzen processors.

```text
zorther/
│
├── config/
│   ├── __init__.py
│   ├── model_config.py        # Model architecture configuration (layers, hidden size, attention, etc.)
│   ├── training_config.py     # Training hyperparameters and optimization settings
│   ├── dataset_config.py      # Dataset paths and preprocessing configuration
│   └── cpu_config.py          # CPU optimization and hardware-specific settings
│
├── tokenizer/
│   ├── __init__.py
│   ├── base.py                # Abstract tokenizer interface
│   ├── bpe_tokenizer.py       # Byte Pair Encoding (BPE) tokenizer implementation
│   ├── sentencepiece_tok.py   # SentencePiece tokenizer support
│   ├── streaming.py           # Streaming tokenization for large datasets
│   └── vocab_stats.py         # Vocabulary statistics and analysis utilities
│
├── model/
│   ├── __init__.py
│   ├── transformer.py         # Core decoder-only Transformer architecture
│   ├── embeddings.py          # Token embeddings and Rotary Positional Embeddings (RoPE)
│   ├── attention.py           # Multi-Head Attention, GQA, MQA, and Flash Attention interface
│   ├── layers.py              # RMSNorm, SwiGLU/GeGLU feed-forward layers, and residual blocks
│   └── cache.py               # Dynamic and memory-aligned KV cache implementation
│
├── data/
│   ├── __init__.py
│   ├── dataset.py             # Dataset loader for JSONL, CSV, Text, and Parquet files
│   ├── preprocessor.py        # Data cleaning, normalization, and MinHash LSH deduplication
│   ├── packing.py             # Sequence packing algorithm for efficient training
│   └── dataloader.py          # Dynamic batching and memory-efficient streaming dataloader
│
├── optimization/
│   ├── __init__.py
│   ├── cpu.py                 # AMD Ryzen optimization (AVX2, OpenMP, thread affinity)
│   ├── optimizer.py           # AdamW optimizer and learning rate schedulers
│   └── loss.py                # Cross-Entropy loss with optional Label Smoothing
│
├── pipeline/
│   ├── __init__.py
│   ├── train.py               # Training and validation pipeline
│   ├── inference.py           # Real-time text generation and sampling engine
│   └── evaluate.py            # Model evaluation and benchmarking pipeline
│
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Console, CSV, and TensorBoard logging utilities
│   ├── checkpoint.py          # Model checkpoint saving and loading (Safetensors & state_dict)
│   └── diagnostics.py         # Hardware diagnostics and CPU performance monitoring
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py          # Transformer architecture unit tests
│   ├── test_tokenizer.py      # Tokenizer validation tests
│   ├── test_data.py           # Dataset and preprocessing tests
│   └── test_optimization.py   # Optimization and performance tests
│
├── dataset/
│   ├── books/                 # Books corpus
│   ├── wikipedia/             # Wikipedia dataset
│   ├── knowledge/             # General knowledge dataset
│   ├── reasoning/             # Logical reasoning dataset
│   ├── mathematics/           # Mathematics dataset
│   ├── coding/                # Programming and code dataset
│   ├── conversation/          # Conversational dataset
│   ├── language/              # Multilingual language dataset
│   └── config/
│       └── dataset_config.json # Dataset configuration file
│
├── requirements.txt           # Project dependencies
├── train.py                   # Main training entry point
├── generate.py                # Text generation entry point
└── README.md                  # Project documentation
```

---

# 🚀 Project Highlights

- Decoder-only Transformer architecture
- Modular and scalable codebase
- Rotary Positional Embeddings (RoPE)
- Multi-Head Attention (MHA)
- Grouped Query Attention (GQA)
- Multi Query Attention (MQA)
- Flash Attention interface
- RMSNorm normalization
- SwiGLU / GeGLU feed-forward network
- Dynamic KV Cache
- Sequence Packing
- Streaming Data Loader
- MinHash LSH Deduplication
- AdamW Optimizer
- Cosine Learning Rate Scheduler
- Warmup Scheduler
- Label Smoothing
- Safetensors Checkpoint Support
- TensorBoard Logging
- CPU Diagnostics
- AMD Ryzen (AVX2 + OpenMP) Optimization
- JSONL, CSV, Parquet, and Text Dataset Support

---

## 🎯 Design Goals

Zorther is designed to provide:

- High-performance Transformer architecture
- Efficient CPU training and inference
- Modular and maintainable source code
- Easy extensibility for future research
- Support for multilingual datasets
- Fast inference with optimized KV caching
- Clean, production-ready project structure

---

## ⚡ Current Status

**Project Name:** Zorther

**Type:** Large Language Model (LLM)

**Architecture:** Decoder-Only Transformer

**Status:** Active Development 🚧