# Learning Projects with AI

Personal learning projects focused on generative AI, PyTorch, and Hugging Face transformers.

Each project has its own isolated Python virtual environment for complete dependency management.

## Quick Start - Create a New Project

```bash
./new_project.sh my_project
cd my_project
source activate.sh
jupyter lab
```

That's it! Your project is ready with all dependencies installed.

## Available Libraries (in each venv)

- **jupyter** / **jupyterlab** - Interactive notebooks
- **torch** / **torchvision** / **torchaudio** - PyTorch deep learning framework
- **transformers** - Hugging Face transformer models
- **datasets** - Hugging Face datasets library

## Project Structure

```
learning_projects/
├── new_project.sh            # Script to create new projects
├── project1/
│   ├── .venv/                # Project 1's isolated environment
│   ├── notebooks/            # Your Jupyter notebooks
│   ├── data/                 # Project data
│   └── activate.sh           # Activation script
├── project2/
│   ├── .venv/                # Project 2's isolated environment
│   ├── notebooks/
│   ├── data/
│   └── activate.sh
└── ...
```

## Usage

### Create a new project
```bash
./new_project.sh my_new_ai_project
```

This automatically:
- Creates a project directory
- Sets up a fresh virtual environment
- Installs all required packages
- Creates a `notebooks/` and `data/` folder structure
- Generates an activation script

### Work on a project
```bash
cd my_project
source activate.sh
jupyter lab
```

### Deactivate environment
```bash
deactivate
```

## Benefits

- ✓ Each project has isolated dependencies (no conflicts)
- ✓ Easy to manage different library versions per project
- ✓ Clean virtual environments with only what you need
- ✓ Simple one-line project creation
