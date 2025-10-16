#!/bin/bash
set -e

ENV_NAME="sport-agent"
REQ_FILE="requirements.txt"
REQ_YML="requirements.yml"
TEMP_YML="temp_env.yml"
TEMP_TXT="temp_env.txt"

echo "🐍 Setting up Conda environment: $ENV_NAME"

# Check if Conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Installing Miniconda..."

    OS_TYPE="$(uname -s)"
    if [[ "$OS_TYPE" == "Linux" ]]; then
        MINICONDA="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        MINICONDA="Miniconda3-latest-MacOSX-arm64.sh"
    else
        echo "❌ Unsupported OS for automatic Conda install."
        exit 1
    fi

    # Download installer
    curl -LO "https://repo.anaconda.com/miniconda/$MINICONDA"

    # Install silently to $HOME/miniconda3
    bash "$MINICONDA" -b -p "$HOME/miniconda3"

    # Add Conda to PATH for this session
    export PATH="$HOME/miniconda3/bin:$PATH"

    # Initialize conda for bash/zsh
    conda init bash 2>/dev/null || conda init zsh 2>/dev/null

    echo "✅ Miniconda installed."
fi

# Check OS type
OS_TYPE="$(uname -s)"
echo "💻 Detected OS: $OS_TYPE"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
else
    # fallback to check pytorch cuda
    GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo False)
fi

echo "🖥 GPU Available: $GPU_AVAILABLE"

# Remove xformers if no GPU
if [[ "$GPU_AVAILABLE" != "True" ]]; then
    echo "⚠️ No GPU detected, removing xformers"

    # Remove xformers line from requirements.yml
    grep -v -E '^\s*-\s*xformers' "$REQ_YML" > "$TEMP_YML"

    # Convert to txt for Linux, yml for others
    if [[ "$OS_TYPE" == "Linux" ]]; then
        TEMP_FILE="$TEMP_TXT"
        # Extract pip packages into txt
        grep -A 100 "pip:" "$TEMP_YML" | sed -n '/pip:/,$p' | sed '1d' > "$TEMP_FILE"
    else
        TEMP_FILE="$TEMP_YML"
    fi
else
    TEMP_FILE="$REQ_FILE"
fi

echo "📦 Creating Conda environment from $TEMP_FILE..."

if [[ "$TEMP_FILE" == *.yml ]]; then
    conda env create --name "$ENV_NAME" -f "$TEMP_FILE"
else
    conda create --name "$ENV_NAME" --file "$TEMP_FILE"
fi

rm "$TEMP_YML" "$TEMP_TXT" 2>/dev/null || true

echo "🚀 Environment $ENV_NAME setup complete!"
echo "To activate: conda activate $ENV_NAME"y
