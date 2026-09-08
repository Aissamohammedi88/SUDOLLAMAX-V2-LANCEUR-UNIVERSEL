#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUDOLLAMAX ULTIMATE v2.0.0 — LANCEUR UNIVERSEL
Auteur : Aissa Mohammedi
Remplace Ollama avec support complet : HF, PyTorch, TF, ONNX, TensorRT, JAX, GGUF
"""

import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DÉTECTION GPU AVANCÉE
# ═══════════════════════════════════════════════════════════════════════════

class GPUDetector:
    @staticmethod
    def detect_gpu() -> Tuple[str, Optional[str]]:
        """Détecte GPU : CUDA (NVIDIA) | ROCm (AMD) | Metal (Apple) | CPU"""
        
        # NVIDIA CUDA
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], 
                                   capture_output=True, timeout=2)
            if result.returncode == 0:
                cuda_version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
                version = cuda_version.stdout.split()[-1] if cuda_version.returncode == 0 else "unknown"
                return "cuda", version
        except:
            pass
        
        # AMD ROCm
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, timeout=2)
            if result.returncode == 0:
                return "rocm", "amd"
        except:
            pass
        
        # Apple Metal
        if sys.platform == "darwin":
            try:
                result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
                if "Apple" in result.stdout or "M1" in result.stdout or "M2" in result.stdout:
                    return "metal", "apple"
            except:
                pass
        
        return "cpu", None

# ═══════════════════════════════════════════════════════════════════════════
# 2. DÉTECTION AVANCÉE DU TYPE DE MODÈLE
# ═══════════════════════════════════════════════════════════════════════════

class ModelDetector:
    SIGNATURES = {
        "huggingface": ["config.json", "model.safetensors", "pytorch_model.bin"],
        "tensorflow": ["saved_model.pb", "keras_metadata.pb"],
        "pytorch": ["pytorch_model.bin", "model.pt", "state_dict.pth"],
        "onnx": ["model.onnx", "*.onnx"],
        "tensorrt": ["model.engine", "*.engine"],
        "jax": ["params.msgpack", "*.msgpack"],
        "gguf": ["*.gguf"],
    }
    
    @staticmethod
    def detect(model_path: Path) -> Tuple[str, float]:
        """Détecte le type avec score de confiance (0-1)"""
        path = Path(model_path)
        
        if not path.exists():
            return "unknown", 0.0
        
        # Fichier unique
        if path.is_file():
            if path.suffix == ".gguf":
                return "gguf", 1.0
            if path.suffix == ".onnx":
                return "onnx", 1.0
            if path.suffix in [".pt", ".pth"]:
                return "pytorch", 0.9
            if path.suffix == ".engine":
                return "tensorrt", 1.0
        
        # Répertoire — scan les fichiers
        if path.is_dir():
            files = {f.name for f in path.rglob("*")}
            scores = {}
            
            for model_type, signatures in ModelDetector.SIGNATURES.items():
                match_count = sum(1 for sig in signatures if sig in files or any(f.endswith(sig.replace("*", "")) for f in files))
                scores[model_type] = match_count / len(signatures)
            
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            if confidence > 0.3:
                return best_type, confidence
        
        return "unknown", 0.0

# ═══════════════════════════════════════════════════════════════════════════
# 3. DÉPENDANCES INTELLIGENTES
# ═══════════════════════════════════════════════════════════════════════════

class DependencyManager:
    ENGINES = {
        "huggingface": ["transformers", "torch", "accelerate"],
        "pytorch": ["torch"],
        "tensorflow": ["tensorflow"],
        "onnx": ["onnxruntime"],
        "tensorrt": ["nvidia-tensorrt"],
        "jax": ["jax", "jaxlib"],
        "gguf": ["llama-cpp-python"],
    }
    
    @staticmethod
    def install_for_model(model_type: str, gpu: str):
        """Installe UNIQUEMENT les dépendances nécessaires"""
        deps = DependencyManager.ENGINES.get(model_type, [])
        
        if not deps:
            logger.warning(f"Aucune dépendance pour {model_type}")
            return
        
        logger.info(f"📦 Installation des dépendances pour {model_type} (GPU: {gpu})")
        
        # Adapter l'installation selon GPU
        pip_args = ["pip", "install", "--upgrade"]
        
        if gpu == "cuda":
            if "torch" in deps:
                deps.remove("torch")
                pip_args.extend(["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
        
        elif gpu == "rocm":
            if "torch" in deps:
                deps.remove("torch")
                pip_args.extend(["torch", "--index-url", "https://download.pytorch.org/whl/rocm5.7"])
        
        elif gpu == "metal":
            if "torch" in deps:
                deps.remove("torch")
                pip_args.append("torch")  # macOS auto-detect
        
        pip_args.extend(deps)
        subprocess.run(pip_args)

# ═══════════════════════════════════════════════════════════════════════════
# 4. LANCEURS PAR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ModelRunner:
    @staticmethod
    def run_huggingface(model_path: Path):
        from transformers import pipeline, AutoModel
        logger.info("🤗 Chargement HuggingFace...")
        try:
            model = AutoModel.from_pretrained(str(model_path))
            logger.info("✅ Modèle HuggingFace chargé")
            return model
        except Exception as e:
            logger.error(f"❌ Erreur HF: {e}")
    
    @staticmethod
    def run_pytorch(model_path: Path):
        import torch
        logger.info("🔥 Chargement PyTorch...")
        try:
            model = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
            logger.info("✅ Modèle PyTorch chargé")
            return model
        except Exception as e:
            logger.error(f"❌ Erreur PyTorch: {e}")
    
    @staticmethod
    def run_tensorflow(model_path: Path):
        import tensorflow as tf
        logger.info("📊 Chargement TensorFlow...")
        try:
            model = tf.saved_model.load(str(model_path))
            logger.info("✅ Modèle TensorFlow chargé")
            return model
        except Exception as e:
            logger.error(f"❌ Erreur TF: {e}")
    
    @staticmethod
    def run_onnx(model_path: Path):
        import onnxruntime as ort
        logger.info("📦 Chargement ONNX...")
        try:
            session = ort.InferenceSession(str(model_path / "model.onnx"))
            logger.info("✅ Modèle ONNX chargé")
            return session
        except Exception as e:
            logger.error(f"❌ Erreur ONNX: {e}")
    
    @staticmethod
    def run_tensorrt(model_path: Path):
        import tensorrt as trt
        logger.info("⚡ Chargement TensorRT...")
        try:
            logger.trt = trt.Logger(trt.Logger.WARNING)
            with open(model_path / "model.engine", "rb") as f:
                runtime = trt.Runtime(logger.trt)
                engine = runtime.deserialize_cuda_engine(f.read())
            logger.info("✅ Modèle TensorRT chargé")
            return engine
        except Exception as e:
            logger.error(f"❌ Erreur TensorRT: {e}")
    
    @staticmethod
    def run_jax(model_path: Path):
        import jax
        import jax.numpy as jnp
        logger.info("🚀 Chargement JAX...")
        try:
            with open(model_path / "params.msgpack", "rb") as f:
                params = f.read()
            logger.info("✅ Modèle JAX chargé")
            return params
        except Exception as e:
            logger.error(f"❌ Erreur JAX: {e}")
    
    @staticmethod
    def run_gguf(model_path: Path):
        import llama_cpp
        logger.info("🦙 Chargement GGUF...")
        try:
            llm = llama_cpp.Llama(str(model_path))
            logger.info("✅ Modèle GGUF chargé")
            return llm
        except Exception as e:
            logger.error(f"❌ Erreur GGUF: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN ORCHESTRATEUR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: sudollamax <model_path_or_name>")
        sys.exit(1)
    
    model_input = sys.argv[1]
    model_path = Path(model_input).expanduser()
    
    # Détection GPU
    gpu_type, gpu_version = GPUDetector.detect_gpu()
    logger.info(f"🖥️  GPU détecté: {gpu_type} ({gpu_version or 'N/A'})")
    
    # Détection modèle
    model_type, confidence = ModelDetector.detect(model_path)
    logger.info(f"🔍 Type détecté: {model_type} (confiance: {confidence:.2f})")
    
    if model_type == "unknown":
        logger.error(f"❌ Type de modèle non identifié")
        sys.exit(1)
    
    # Installation des dépendances
    DependencyManager.install_for_model(model_type, gpu_type)
    
    # Lancement du modèle
    runners = {
        "huggingface": ModelRunner.run_huggingface,
        "pytorch": ModelRunner.run_pytorch,
        "tensorflow": ModelRunner.run_tensorflow,
        "onnx": ModelRunner.run_onnx,
        "tensorrt": ModelRunner.run_tensorrt,
        "jax": ModelRunner.run_jax,
        "gguf": ModelRunner.run_gguf,
    }
    
    runner = runners.get(model_type)
    if runner:
        model = runner(model_path)
        logger.info("🚀 Modèle prêt à l'utilisation")
    else:
        logger.error(f"❌ Runner non trouvé pour {model_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()