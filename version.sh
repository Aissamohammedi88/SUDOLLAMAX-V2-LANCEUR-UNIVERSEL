#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
# SUDOLLAMAX ULTIMATE v2.2.0 — CLIENT UNIVERSEL + MENU INTERACTIF
# Auteur : Aissa Mohammedi (DGK) — Nexus Quantum
#
# v2.2 : syntaxe bash corrigee + menu list/run/pull (Ollama + HuggingFace)
#        Additif : ne reinstalle jamais ce qui existe deja.
# ════════════════════════════════════════════════════════════════════════
set -uo pipefail   # PAS de -e : on gere chaque echec

# ─── Couleurs ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
BLUE='\033[0;34m';  CYAN='\033[0;36m';  MAG='\033[0;35m'; NC='\033[0m'

info()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()    { echo -e "${RED}[ERREUR]${NC} $*"; }
section() { echo -e "\n${BLUE}===== $* =====${NC}"; }
title()   { echo -e "\n${MAG}--- $* ---${NC}"; }

# ─── Dossiers ────────────────────────────────────────────────────────────
BASE_DIR="${HOME}/sudollamax_ultimate"
STACKS_DIR="${BASE_DIR}/stacks"
MODELS_DIR="${BASE_DIR}/modeles"        # OLLAMA_MODELS
HF_DIR="${BASE_DIR}/hf_models"          # modèles HF téléchargés
LOGS_DIR="${BASE_DIR}/logs"
BIN_DIR="${HOME}/.local/bin"
DEFAULT_MODEL="qwen2.5:1.5b"

mkdir -p "$BASE_DIR" "$STACKS_DIR" "$MODELS_DIR" "$HF_DIR" "$LOGS_DIR" "$BIN_DIR"

# ─── 0. Détection GPU ────────────────────────────────────────────────────
detect_gpu() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        echo "nvidia"
    elif [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
        echo "apple"
    else
        echo "cpu"
    fi
}
GPU_MODE="$(detect_gpu)"

# ─── 1. Dépendances système ──────────────────────────────────────────────
install_system_deps() {
    section "1. DÉPENDANCES SYSTÈME"
    sudo apt-get update -qq || warn "apt update échoué (on continue)"
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev \
        build-essential git curl wget cmake g++ \
        libopenblas-dev libssl-dev pkg-config git-lfs \
        || { fail "apt install échoué"; return 1; }
    info "Dépendances système installées"

    if [[ "$GPU_MODE" == "nvidia" ]]; then
        warn "CUDA toolkit requis pour TensorRT/PyTorch GPU (~3 Go)"
        sudo apt-get install -y -qq nvidia-cuda-toolkit || warn "CUDA toolkit échoué (continue CPU)"
    else
        info "Pas de GPU NVIDIA — skip nvidia-cuda-toolkit (économie 3 Go)"
    fi
}

# ─── 2. Ollama ───────────────────────────────────────────────────────────
install_ollama() {
    section "2. OLLAMA"
    if ! command -v ollama &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh || { fail "Ollama échoué"; return 1; }
    else
        info "Ollama déjà installé : $(command -v ollama)"
    fi
    # Pointer OLLAMA_MODELS vers notre dossier (additif dans .bashrc)
    if ! grep -q "OLLAMA_MODELS" ~/.bashrc 2>/dev/null; then
        echo "export OLLAMA_MODELS=\"${MODELS_DIR}\"" >> ~/.bashrc
    fi
    # Démarrer le service Ollama si pas actif
    if ! pgrep -x ollama &>/dev/null; then
        nohup ollama serve >"${LOGS_DIR}/ollama.log" 2>&1 &
        sleep 2
    fi
    info "Ollama prêt — modèles dans ${MODELS_DIR}"
}

# ─── 3. Venvs isolés (le vrai fix du conflit TF/PyTorch) ──────────────────
make_venv() {
    local name="$1"
    local dir="${STACKS_DIR}/venv-${name}"
    if [[ ! -d "$dir" ]]; then
        python3 -m venv "$dir" || { fail "venv ${name} échoué"; return 1; }
    fi
    # shellcheck disable=SC1091
    source "${dir}/bin/activate"
    pip install -q --upgrade pip setuptools wheel
    deactivate
    info "venv-${name} prêt"
}

install_venvs() {
    section "3. VENVS ISOLÉS (main / torch / tf)"
    make_venv main
    make_venv torch
    make_venv tf

    # main : transformers + huggingface-cli
    source "${STACKS_DIR}/venv-main/bin/activate"
    pip install -q --upgrade transformers huggingface_hub accelerate sentencepiece \
        || warn "pip main incomplet"
    deactivate

    # torch : pytorch CPU ou CUDA
    source "${STACKS_DIR}/venv-torch/bin/activate"
    if [[ "$GPU_MODE" == "nvidia" ]]; then
        pip install -q --upgrade torch torchvision --index-url \
            https://download.pytorch.org/whl/cu121 || warn "torch CUDA échoué"
    else
        pip install -q --upgrade torch torchvision --index-url \
            https://download.pytorch.org/whl/cpu || warn "torch CPU échoué"
    fi
    pip install -q --upgrade transformers accelerate sentencepiece \
        || warn "transformers dans venv-torch incomplet"
    deactivate

    # tf : tensorflow + onnx + tensorrt (optionnel)
    source "${STACKS_DIR}/venv-tf/bin/activate"
    pip install -q --upgrade tensorflow onnxruntime \
        || warn "tf/onnx dans venv-tf (tensorrt skip si GPU absent)"
    deactivate

    info "Venvs populés"
}

# ─── 7. Lanceur Python sudollamax.py ──────────────────────────────────────
write_python_launcher() {
    section "7. LANCEUR PYTHON sudollamax.py"
    cat > "${BASE_DIR}/sudollamax.py" << 'PYEOF'
#!/usr/bin/env python3
"""SUDOLLAMAX — Lanceur universel de modèles.
Détecte le type de modèle et route vers le bon runtime."""
import sys, subprocess
from pathlib import Path

GREEN=f"\033[0;32m"; YELLOW=f"\033[0;33m"; RED=f"\033[0;31m"
CYAN=f"\033[0;36m"; NC=f"\033[0m"
def info(m): print(f"{GREEN}[OK]{NC} {m}")
def warn(m): print(f"{YELLOW}[WARN]{NC} {m}")
def fail(m): print(f"{RED}[ERR]{NC} {m}")

def detect_model_type(model_path: str) -> str:
    p = Path(model_path).expanduser()
    # Ollama model name (e.g. qwen2.5:1.5b) — pas un chemin fichier
    if not p.exists() and ":" in model_path and "/" not in model_path:
        return "ollama"
    if p.is_dir():
        if (p/"model.engine").exists():     return "tensorrt"
        if (p/"model.onnx").exists():       return "onnx"
        if (p/"saved_model.pb").exists():   return "tensorflow"
        if (p/"config.json").exists() or (p/"pytorch_model.bin").exists() \
           or (p/"model.pt").exists() or (p/"model.safetensors").exists():
            return "huggingface"
    if p.suffix == ".gguf":                 return "ollama"
    if p.is_file() and "gguf" in p.name.lower(): return "ollama"
    return "unknown"

def run_ollama(model_path: str, prompt: str = "Hello"):
    subprocess.run(["ollama", "run", str(model_path), prompt], check=False)

def run_huggingface(model_path: str, prompt: str = "Hello"):
    from transformers import pipeline
    info("Chargement HuggingFace Transformers…")
    pipe = pipeline("text-generation", model=str(Path(model_path).expanduser()))
    out = pipe(prompt, max_new_tokens=50)
    print(out[0]["generated_text"])

def run_pytorch(model_path: str, prompt: str = ""):
    import torch
    p = Path(model_path).expanduser()
    info("Chargement PyTorch natif…")
    weights = p / "pytorch_model.bin"
    if not weights.exists(): weights = p / "model.pt"
    model = torch.load(weights, map_location="cpu")
    print(f"Modèle chargé : {model.__class__.__name__}")
    if prompt: warn(f"Prompt ignoré (pas de tokenizer) : {prompt!r}")

def run_tensorflow(model_path: str, prompt: str = ""):
    import tensorflow as tf
    p = Path(model_path).expanduser()
    info("Chargement TensorFlow SavedModel…")
    model = tf.saved_model.load(str(p))
    print(f"Signatures : {list(model.signatures.keys())}")

def run_onnx(model_path: str, prompt: str = ""):
    import onnxruntime as ort
    p = Path(model_path).expanduser()
    info("Chargement ONNX Runtime…")
    session = ort.InferenceSession(str(p / "model.onnx"))
    print(f"Entrées : {[i.name for i in session.get_inputs()]}")

def run_tensorrt(model_path: str, prompt: str = ""):
    import tensorrt as trt
    p = Path(model_path).expanduser()
    info("Chargement TensorRT Engine…")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(p / "model.engine", "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    print(f"Moteur TensorRT chargé : {'OK' if engine else 'FAIL'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: sudollamax <model_path_or_name> [prompt]"); sys.exit(1)
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    mtype = detect_model_type(model_path)
    print(f"\033[1;36m[SUDOLLAMAX]\033[0m Type : \033[1;32m{mtype}\033[0m")
    runners = {"ollama": run_ollama, "huggingface": run_huggingface,
               "pytorch": run_pytorch, "tensorflow": run_tensorflow,
               "onnx": run_onnx, "tensorrt": run_tensorrt}
    runner = runners.get(mtype)
    if runner: runner(model_path, prompt)
    else:
        fail(f"Format non supporté : {mtype}")
        sys.exit(2)

if __name__ == "__main__":
    main()
PYEOF
    chmod +x "${BASE_DIR}/sudollamax.py"
    info "Lanceur Python écrit : ${BASE_DIR}/sudollamax.py"
}

# ─── 8. Wrapper bash global ───────────────────────────────────────────────
write_bash_wrapper() {
    section "8. WRAPPER GLOBAL ${BIN_DIR}/sudollamax"
    cat > "${BIN_DIR}/sudollamax" << 'WRAPPER'
#!/usr/bin/env bash
# Route vers le bon venv selon le type de modèle
BASE_DIR="${HOME}/sudollamax_ultimate"
STACKS="${BASE_DIR}/stacks"
MODEL="${1:-}"; PROMPT="${2:-Hello}"

if [[ -z "$MODEL" ]]; then
    echo "Usage: sudollamax <model> [prompt]"
    echo "       sudollamax --menu    # menu interactif"
    echo "       sudollamax --status  # statut système"
    exit 1
fi
if [[ "$MODEL" == "--menu" ]]; then
    exec bash "${BASE_DIR}/sudollamax_ultimate.sh" --menu
fi
if [[ "$MODEL" == "--status" ]]; then
    exec bash "${BASE_DIR}/sudollamax_ultimate.sh" --status
fi

# Détection rapide du type (Ollama / torch / tensorflow / onnx)
TYPE=$(python3 - <<PY 2>/dev/null || echo "unknown"
from pathlib import Path
m = "${MODEL}"
p = Path(m).expanduser()
if not p.exists() and ":" in m and "/" not in m:
    print("ollama"); raise SystemExit
if p.is_dir():
    if (p/"saved_model.pb").exists():        print("tensorflow"); raise SystemExit
    if (p/"config.json").exists() or (p/"pytorch_model.bin").exists() \
       or (p/"model.safetensors").exists() or (p/"model.pt").exists():
        print("torch"); raise SystemExit
    if (p/"model.onnx").exists() or (p/"model.engine").exists():
        print("torch"); raise SystemExit
if p.suffix == ".gguf" or "gguf" in p.name.lower():
    print("ollama"); raise SystemExit
print("unknown")
PY
)
case "$TYPE" in
    ollama)
        exec ollama run "$MODEL" "$PROMPT" ;;
    torch)
        source "${STACKS}/venv-torch/bin/activate" 2>/dev/null
        exec python3 "${BASE_DIR}/sudollamax.py" "$MODEL" "$PROMPT" ;;
    tensorflow)
        source "${STACKS}/venv-tf/bin/activate" 2>/dev/null
        exec python3 "${BASE_DIR}/sudollamax.py" "$MODEL" "$PROMPT" ;;
    *)
        source "${STACKS}/venv-main/bin/activate" 2>/dev/null
        exec python3 "${BASE_DIR}/sudollamax.py" "$MODEL" "$PROMPT" ;;
esac
WRAPPER
    chmod +x "${BIN_DIR}/sudollamax"
    # Ajouter ~/.local/bin au PATH si nécessaire
    if ! grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    info "Wrapper : ${BIN_DIR}/sudollamax"
}

# ─── Catalogue HuggingFace (modèles populaires pour pull) ─────────────────
HF_CATALOG=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    "microsoft/Phi-3.5-mini-instruct"
    "microsoft/Phi-4-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Ministral-3B-Instruct-2410"
    "mistralai/Ministral-8B-Instruct-2410"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "google/gemma-3-4b-it"
    "HuggingFaceH4/zephyr-7b-beta"
    "teknium/OpenHermes-2.5-Mistral-7B"
    "tinyllama/TinyLlama-1.1B-Chat-v1.0"
    "bigcode/starcoder2-3b"
    "bigcode/starcoder2-7b"
    "TheBloke/Llama-2-7B-Chat-AWQ"
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    "bartowski/Qwen2.5-7B-Instruct-GGUF"
    "second-state/Qwen2.5-Coder-7B-Instruct-GGUF"
    "unsloth/llama-3-8b-Instruct-bnb-4bit"
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
)

# ─── Menu : Lister modèles Ollama ─────────────────────────────────────────
list_ollama() {
    title "MODÈLES OLLAMA LOCAUX"
    if ! command -v ollama &>/dev/null; then fail "Ollama absent"; return; fi
    local list
    list=$(ollama list 2>/dev/null | tail -n +2)
    if [[ -z "$list" ]]; then warn "Aucun modèle Ollama"; return; fi
    ollama list
}

# ─── Menu : Lister modèles HuggingFace ────────────────────────────────────
list_hf() {
    title "MODÈLES HUGGINGFACE LOCAUX"
    if [[ ! -d "$HF_DIR" ]]; then warn "Dossier $HF_DIR vide"; return; fi
    local found=0
    while IFS= read -r d; do
        if [[ -f "$d/config.json" ]] || [[ -f "$d/pytorch_model.bin" ]] \
           || [[ -f "$d/model.safetensors" ]] || [[ -f "$d/model.onnx" ]]; then
            echo -e "  ${CYAN}$(basename "$d")${NC}  ->  $d"
            found=$((found+1))
        fi
    done < <(find "$HF_DIR" -maxdepth 2 -type d 2>/dev/null)
    [[ $found -eq 0 ]] && warn "Aucun modèle HF dans $HF_DIR"
    info "$found modèle(s) HF"
}

# ─── Menu : Run Ollama ────────────────────────────────────────────────────
run_ollama_interactive() {
    list_ollama
    echo
    read -rp "Modèle à lancer [${DEFAULT_MODEL}] : " m
    m="${m:-$DEFAULT_MODEL}"
    read -rp "Prompt (Enter = chat interactif) : " p
    if [[ -z "$p" ]]; then
        ollama run "$m"
    else
        ollama run "$m" "$p"
    fi
}

# ─── Menu : Run HuggingFace ───────────────────────────────────────────────
run_hf_interactive() {
    list_hf
    echo
    read -rp "Chemin du modèle HF : " m
    [[ -z "$m" ]] && { warn "Annulé"; return; }
    m=$(eval echo "$m")
    read -rp "Prompt [Hello] : " p
    p="${p:-Hello}"
    "${BIN_DIR}/sudollamax" "$m" "$p"
}

# ─── Menu : Pull HuggingFace (catalogue) ──────────────────────────────────
pull_hf_catalog() {
    title "CATALOGUE HUGGINGFACE — PULL"
    echo "Choisissez un modèle (1-${#HF_CATALOG[@]}) ou 'q' pour annuler :"
    echo
    for i in "${!HF_CATALOG[@]}"; do
        printf "  %2d) %s\n" "$((i+1))" "${HF_CATALOG[$i]}"
    done
    echo
    read -rp "Choix > " c
    [[ "$c" == "q" ]] && return
    if ! [[ "$c" =~ ^[0-9]+$ ]] || (( c < 1 || c > ${#HF_CATALOG[@]} )); then
        fail "Choix invalide"; return
    fi
    local repo="${HF_CATALOG[$((c-1))]}"
    local dest="${HF_DIR}/$(basename "$repo")"
    info "Pull HF : $repo -> $dest"

    # Activer venv-main qui contient huggingface_hub
    if [[ -f "${STACKS_DIR}/venv-main/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${STACKS_DIR}/venv-main/bin/activate"
    fi
    if ! command -v huggingface-cli &>/dev/null; then
        warn "huggingface-cli absent — installation dans venv-main"
        pip install -q huggingface_hub || { fail "pip install échoué"; return; }
    fi
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$repo" --local-dir "$dest" || fail "Téléchargement échoué"
    else
        # fallback git clone avec LFS
        git lfs install 2>/dev/null || sudo apt-get install -y -qq git-lfs
        git clone "https://huggingface.co/${repo}" "$dest" || fail "git clone échoué"
    fi
    deactivate 2>/dev/null || true
    info "Téléchargement terminé : $dest"
}

# ─── Menu : Pull Ollama ───────────────────────────────────────────────────
pull_ollama_interactive() {
    title "PULL OLLAMA"
    echo "Modèles populaires :"
    echo "  qwen2.5:0.5b   qwen2.5:1.5b   qwen2.5:3b   qwen2.5:7b"
    echo "  qwen2.5-coder:1.5b   qwen2.5-coder:7b"
    echo "  llama3.2:1b    llama3.2:3b    llama3.1:8b"
    echo "  mistral:7b     mistral-nemo:12b"
    echo "  gemma2:2b      gemma2:9b"
    echo "  phi3:mini      phi3.5:mini"
    echo "  tinyllama:1.1b"
    echo
    read -rp "Modèle à pull [qwen2.5:1.5b] : " m
    m="${m:-qwen2.5:1.5b}"
    ollama pull "$m" || fail "pull échoué"
}

# ─── Menu : Statut système ────────────────────────────────────────────────
status_report() {
    title "STATUT SYSTÈME"
    echo -e "  GPU           : ${CYAN}${GPU_MODE}${NC}"
    if command -v ollama &>/dev/null; then
        echo -e "  Ollama        : ${GREEN}OK${NC} ($(command -v ollama))"
        local n
        n=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
        echo -e "  Modèles Ollama: ${n}"
    else
        echo -e "  Ollama        : ${RED}ABSENT${NC}"
    fi
    for v in main torch tf; do
        if [[ -f "${STACKS_DIR}/venv-${v}/bin/activate" ]]; then
            echo -e "  venv-${v}      : ${GREEN}OK${NC}"
        else
            echo -e "  venv-${v}      : ${YELLOW}MANQUANT${NC}"
        fi
    done
    echo -e "  Base          : ${BASE_DIR}"
    echo -e "  HF dir        : ${HF_DIR}"
    echo -e "  Wrapper       : ${BIN_DIR}/sudollamax"
    echo -e "  Modèle défaut : ${DEFAULT_MODEL}"
}

# ─── Menu principal ───────────────────────────────────────────────────────
menu() {
    while true; do
        title "SUDOLLAMAX MENU — v2.2.0"
        echo "  1) Lister modèles Ollama locaux"
        echo "  2) Lister modèles HuggingFace locaux"
        echo "  3) Run modèle Ollama"
        echo "  4) Run modèle HuggingFace"
        echo "  5) Pull modèle HuggingFace (catalogue)"
        echo "  6) Pull modèle Ollama"
        echo "  7) Statut système"
        echo "  8) Quitter"
        echo
        read -rp "Choix > " choice
        case "$choice" in
            1) list_ollama ;;
            2) list_hf ;;
            3) run_ollama_interactive ;;
            4) run_hf_interactive ;;
            5) pull_hf_catalog ;;
            6) pull_ollama_interactive ;;
            7) status_report ;;
            8) info "Au revoir"; break ;;
            *) warn "Choix invalide" ;;
        esac
    done
}

# ─── 9. Self-test ─────────────────────────────────────────────────────────
self_test() {
    section "9. TEST"
    if command -v ollama &>/dev/null; then
        if ! ollama list 2>/dev/null | grep -q "^qwen2.5"; then
            warn "qwen2.5 absent — pull qwen2.5:1.5b recommandé (menu choix 6)"
        else
            info "Test run (non-bloquant) :"
            ollama run qwen2.5:1.5b "dis bonjour en 5 mots" 2>&1 | head -3 || warn "test échoué"
        fi
    else
        warn "Ollama absent — test sauté"
    fi
}

# ─── Rapport final ────────────────────────────────────────────────────────
final_report() {
    section "RAPPORT — SUDOLLAMAX ULTIMATE v2.2.0"
    echo -e "  Base      : ${BASE_DIR}"
    echo -e "  Venvs     : ${STACKS_DIR}/venv-{main,torch,tf}"
    echo -e "  Modèles   : ${MODELS_DIR} (OLLAMA_MODELS)"
    echo -e "  HF dir    : ${HF_DIR}"
    echo -e "  Wrapper   : ${BIN_DIR}/sudollamax"
    echo -e "  GPU       : ${CYAN}${GPU_MODE}${NC}"
    echo -e "  Usage     : sudollamax <model> [prompt]"
    echo -e "              sudollamax --menu    (menu interactif)"
    echo -e "              sudollamax --status  (statut)"
    echo -e "  État      : ${GREEN}Client universel prêt${NC}"
}

# ─── Point d'entrée ───────────────────────────────────────────────────────
main() {
    section "SUDOLLAMAX ULTIMATE v2.2.0"
    info "GPU détecté : ${CYAN}${GPU_MODE}${NC}"
    info "Base : ${BASE_DIR}"

    case "${1:-}" in
        --menu)
            menu
            ;;
        --status)
            status_report
            ;;
        --list)
            list_ollama
            list_hf
            ;;
        --install)
            install_system_deps || exit 1
            install_ollama      || exit 1
            install_venvs
            write_python_launcher
            write_bash_wrapper
            self_test
            final_report
            ;;
        --run)
            shift
            "${BIN_DIR}/sudollamax" "$@"
            ;;
        ""|*)
            # Comportement par défaut : si déjà installé -> menu, sinon install complet
            if [[ -f "${BIN_DIR}/sudollamax" ]] && [[ -f "${BASE_DIR}/sudollamax.py" ]]; then
                menu
            else
                install_system_deps || exit 1
                install_ollama      || exit 1
                install_venvs
                write_python_launcher
                write_bash_wrapper
                self_test
                final_report
            fi
            ;;
    esac
}

main "$@"
