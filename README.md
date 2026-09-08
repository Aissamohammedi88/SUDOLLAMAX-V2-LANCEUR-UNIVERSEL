# 🚀 SUDOLLAMAX v2 — Universal AI Model Launcher

**Remplace Ollama avec support complet de tous les frameworks IA**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://www.python.org/downloads/)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20Metal-orange)]()

## 🎯 Objectif

Une **infrastructure IA universelle** qui détecte, installe et lance automatiquement **n'importe quel modèle** sans configuration manuelle.

### ✨ Ce qui fait SUDOLLAMAX différent d'Ollama

| Feature | Ollama | SUDOLLAMAX v2 |
|---------|--------|----------------|
| **Formats supportés** | GGUF seulement | HF, PyTorch, TF, ONNX, TensorRT, JAX, GGUF |
| **Auto-détection GPU** | ❌ | ✅ CUDA/ROCm/Metal |
| **Installation auto des dépendances** | ❌ | ✅ Intelligente par engine |
| **Support TensorRT** | ❌ | ✅ Optimisation NVIDIA native |
| **Support JAX** | ❌ | ✅ Calcul haute performance |
| **Configuration requise** | Manuelle | ❌ Zéro config |
| **Taille footprint** | 2 GB | 5.5 GB (tous les engines) |

## 🚀 Installation

### Pré-requis
- Python 3.8+
- pip
- Git
- (Optionnel) CUDA 11.8+ pour NVIDIA
- (Optionnel) ROCm 5.7+ pour AMD

### Quick Start

```bash
# 1. Cloner le repo
git clone https://github.com/Aissamohammedi88/SUDOLLAMAX-V2-LANCEUR-UNIVERSEL.git
cd SUDOLLAMAX-V2-LANCEUR-UNIVERSEL

# 2. Créer venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows

# 3. Installer SUDOLLAMAX
pip install -r requirements.txt

# 4. Utiliser
python sudollamax.py /path/to/model
```

## 📚 Commandes

```bash
# Lancer un modèle HuggingFace
sudollamax ~/models/llama-7b

# Lancer un modèle GGUF
sudollamax ~/models/neural-chat.gguf

# Lancer un modèle PyTorch
sudollamax ~/models/custom-pytorch-model

# Lancer un modèle TensorFlow
sudollamax ~/models/saved_model

# Lancer un modèle ONNX
sudollamax ~/models/model.onnx

# Lancer un modèle TensorRT
sudollamax ~/models/optimized.engine

# Lancer un modèle JAX
sudollamax ~/models/jax-params
```

## 🔧 Fonctionnalités Avancées

### Auto-Détection GPU

SUDOLLAMAX détecte automatiquement :
- ✅ NVIDIA CUDA (avec version détectée)
- ✅ AMD ROCm
- ✅ Apple Metal (macOS M1/M2/M3)
- ✅ CPU fallback

### Auto-Détection Modèles

Détecte le type de modèle par :
- Structure de répertoires
- Signatures de fichiers
- Score de confiance

### Installation Intelligente des Dépendances

Il installe **SEULEMENT** ce qui est nécessaire :
- PyTorch optimisé pour ton GPU
- TensorFlow adapté
- Dépendances minimales

## 📊 Architecture Interne

```
SUDOLLAMAX
├── GPUDetector
│   ├── CUDA Detection
│   ├── ROCm Detection
│   ├── Metal Detection (macOS)
│   └── CPU Fallback
├── ModelDetector
│   ├── File signature matching
│   ├── Confidence scoring
│   └── Multi-format support
├── DependencyManager
│   ├── Auto-install by engine
│   ├── GPU-specific optimization
│   └── Minimal footprint
└── ModelRunner
    ├── HuggingFace runner
    ├── PyTorch runner
    ├── TensorFlow runner
    ├── ONNX runner
    ├── TensorRT runner
    ├── JAX runner
    └── GGUF runner
```

## 🎓 Exemples Live

### Exemple 1: Lancer un modèle HuggingFace

```bash
$ python sudollamax.py ~/models/llama-7b

🖥️  GPU détecté: cuda (11.8)
🔍 Type détecté: huggingface (confiance: 0.95)
📦 Installation des dépendances pour huggingface (GPU: cuda)
🤗 Chargement HuggingFace...
✅ Modèle HuggingFace chargé
🚀 Modèle prêt à l'utilisation
```

### Exemple 2: Lancer un modèle GGUF

```bash
$ python sudollamax.py ~/models/neural-chat.gguf

🖥️  GPU détecté: cpu (N/A)
🔍 Type détecté: gguf (confiance: 1.00)
📦 Installation des dépendances pour gguf (GPU: cpu)
🦙 Chargement GGUF...
✅ Modèle GGUF chargé
🚀 Modèle prêt à l'utilisation
```

### Exemple 3: Lancer un modèle TensorRT (NVIDIA)

```bash
$ python sudollamax.py ~/models/optimized.engine

🖥️  GPU détecté: cuda (12.0)
🔍 Type détecté: tensorrt (confiance: 1.00)
📦 Installation des dépendances pour tensorrt (GPU: cuda)
⚡ Chargement TensorRT...
✅ Modèle TensorRT chargé
🚀 Modèle prêt à l'utilisation
```

## 🔐 Licence

GPL v3 — Libre et open-source

## 👨‍💻 Auteur

**Aissa Mohammedi**
- GitHub: [@Aissamohammedi88](https://github.com/Aissamohammedi88)
- Project: [SUDOLLAMAX v2](https://github.com/Aissamohammedi88/SUDOLLAMAX-V2-LANCEUR-UNIVERSEL)

---

**SUDOLLAMAX v2** — *Infrastructure IA Universelle* 🚀