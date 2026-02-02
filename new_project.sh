#!/bin/bash
# Create a new project with its own virtual environment

if [ -z "$1" ]; then
    echo "Usage: ./new_project.sh <project_name>"
    echo "Example: ./new_project.sh my_ai_project"
    exit 1
fi

PROJECT_NAME=$1
PROJECT_DIR="/Users/yan.almeida/gen_ai_udacity_learning_projects/$PROJECT_NAME"

# Check if project already exists
if [ -d "$PROJECT_DIR" ]; then
    echo "Error: Project '$PROJECT_NAME' already exists at $PROJECT_DIR"
    exit 1
fi

# Create project directory
echo "Creating project directory: $PROJECT_NAME"
mkdir -p "$PROJECT_DIR"

# Create virtual environment
echo "Setting up virtual environment..."
python3 -m venv "$PROJECT_DIR/.venv"

# Activate and install packages
echo "Installing packages..."
source "$PROJECT_DIR/.venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install jupyter notebook jupyterlab torch torchvision torchaudio transformers datasets

# Create project structure
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/notebooks"

# Create activation script
cat > "$PROJECT_DIR/activate.sh" << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "✓ Virtual environment activated for this project"
echo "✓ Ready to use: jupyter, torch, transformers, datasets"
echo ""
echo "To start Jupyter Lab: jupyter lab"
echo "To start Jupyter Notebook: jupyter notebook"
EOF
chmod +x "$PROJECT_DIR/activate.sh"

# Create a sample notebook
cat > "$PROJECT_DIR/notebooks/example.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Project Setup Verification\n",
    "Run this cell to verify all libraries are installed correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import transformers\n",
    "import datasets\n",
    "\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"Transformers version: {transformers.__version__}\")\n",
    "print(f\"Datasets version: {datasets.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "print(\"\\n✓ All libraries loaded successfully!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo ""
echo "✓ Project created successfully!"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. source activate.sh"
echo "3. jupyter lab"
