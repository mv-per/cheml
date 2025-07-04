#!/bin/bash
# Script to merge base environment.yml with development environment.dev.yml
# and create a conda environment from the merged file

INSTALL_BASE=false
CREATE_ENV=true

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --install-base)
      INSTALL_BASE=true
      CREATE_ENV=false
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --install-base       Install dependencies on base environment (useful for CI/CD)"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

set -e  # Exit immediately if a command exits with a non-zero status

# Check if running with sudo and provide a warning
if [ -n "$SUDO_USER" ]; then
    echo "WARNING: This script is being run with sudo, which can cause issues with environment managers."
    echo "It's recommended to run this script without sudo unless absolutely necessary."
    echo "Continuing with sudo, but this may lead to errors...\n"
fi

# Define paths - ensure we're pointing to the deep-learning directory
echo "Running script from: $(pwd)"
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project root directory: $PROJECT_ROOT"
BASE_ENV_FILE="${PROJECT_ROOT}/environment.yml"
DEV_ENV_FILE="${PROJECT_ROOT}/environment.dev.yml"
OUTPUT_FILE="${PROJECT_ROOT}/environment.merged.yml"

# Check if files exist
if [ ! -f "$BASE_ENV_FILE" ]; then
    echo "Error: Base environment file does not exist: $BASE_ENV_FILE"
    exit 1
fi

if [ ! -f "$DEV_ENV_FILE" ]; then
    echo "Error: Development environment file does not exist: $DEV_ENV_FILE"
    exit 1
fi

find_env_manager() {
    # First check if we're in an active conda environment
    if [ -n "$CONDA_PREFIX" ]; then
        if [ -x "$CONDA_PREFIX/bin/conda" ]; then
            echo "conda"
            return
        elif [ -x "$CONDA_PREFIX/bin/mamba" ]; then
            echo "mamba"
            return
        elif [ -x "$CONDA_PREFIX/bin/micromamba" ]; then
            echo "micromamba"
            return
        fi
    fi

    # Then look in PATH
    if command -v conda &> /dev/null; then
        echo "conda"
    elif command -v micromamba &> /dev/null; then
        echo "micromamba"
    elif command -v mamba &> /dev/null; then
        echo "mamba"
    else
        # Try to source conda initialization if it exists
        local CONDA_PATHS=(
            "$HOME/miniconda3/etc/profile.d/conda.sh"
            "$HOME/anaconda3/etc/profile.d/conda.sh"
            "/opt/conda/etc/profile.d/conda.sh"
            "/usr/local/anaconda3/etc/profile.d/conda.sh"
        )

        for conda_path in "${CONDA_PATHS[@]}"; do
            if [ -f "$conda_path" ]; then
                echo "Sourcing conda from $conda_path"
                source "$conda_path"
                if command -v conda &> /dev/null; then
                    echo "conda"
                    return
                fi
            fi
        done

        # Try to find micromamba in common locations
        local MICROMAMBA_PATHS=(
            "$HOME/.micromamba/bin/micromamba"
            "$HOME/.local/bin/micromamba"
        )

        for micromamba_path in "${MICROMAMBA_PATHS[@]}"; do
            if [ -x "$micromamba_path" ]; then
                echo "$micromamba_path"
                return
            fi
        done

        echo ""  # Not found
    fi
}


ENV_MANAGER=$(find_env_manager)

$ENV_MANAGER install pyyaml -y

echo "Merging environment files..."

# Use Python to merge the files
echo "Using Python to merge environment files..."
python3 -c "
import yaml
import sys
import os

try:
    # Read base environment file
    print('Reading:', os.path.abspath('$BASE_ENV_FILE'), file=sys.stderr)
    with open('$BASE_ENV_FILE', 'r') as f:
        base_env = yaml.safe_load(f)

    # Read development environment file
    print('Reading:', os.path.abspath('$DEV_ENV_FILE'), file=sys.stderr)
    with open('$DEV_ENV_FILE', 'r') as f:
        dev_env = yaml.safe_load(f)

    # Create merged environment
    merged_env = base_env.copy()

    # Track conda dependencies by name (without version) to avoid duplicates
    base_conda_deps = set()
    for dep in base_env.get('dependencies', []):
        if isinstance(dep, str) and '=' in dep:
            base_conda_deps.add(dep.split('=')[0])
        elif isinstance(dep, str):
            base_conda_deps.add(dep)

    # Add dev conda dependencies that don't exist in base
    for dep in dev_env.get('dependencies', []):
        if isinstance(dep, dict) and 'pip' in dep:
            # Handle pip dependencies separately
            continue

        if isinstance(dep, str) and '=' in dep:
            dep_name = dep.split('=')[0]
            if dep_name not in base_conda_deps:
                merged_env['dependencies'].append(dep)
                base_conda_deps.add(dep_name)
        elif isinstance(dep, str) and dep not in base_conda_deps:
            merged_env['dependencies'].append(dep)
            base_conda_deps.add(dep)

    # Merge pip dependencies from both files
    base_pip_deps = {}
    dev_pip_deps = {}
    pip_index = None

    # Find pip dependencies in base environment
    for i, dep in enumerate(base_env.get('dependencies', [])):
        if isinstance(dep, dict) and 'pip' in dep:
            pip_index = i
            for pip_dep in dep['pip']:
                if isinstance(pip_dep, str):
                    pip_name = pip_dep.split('==')[0].split('>=')[0].split('=')[0].strip(\"'[]\")
                    if pip_name.startswith('mlflow'):
                        pip_name = 'mlflow'
                    base_pip_deps[pip_name] = pip_dep

    # Gather pip dependencies from dev environment
    for dep in dev_env.get('dependencies', []):
        if isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                if isinstance(pip_dep, str):
                    pip_name = pip_dep.split('==')[0].split('>=')[0].split('=')[0].strip(\"'[]\")
                    if pip_name.startswith('mlflow'):
                        pip_name = 'mlflow'
                    dev_pip_deps[pip_name] = pip_dep

    # Merge pip dependencies
    for pkg_name, pip_dep in dev_pip_deps.items():
        if pkg_name not in base_pip_deps:
            base_pip_deps[pkg_name] = pip_dep

    # Update pip dependencies in merged environment
    if pip_index is not None:
        merged_env['dependencies'][pip_index]['pip'] = list(base_pip_deps.values())
    elif dev_pip_deps:
        # If no pip section exists in base env but exists in dev
        merged_env['dependencies'].append({'pip': list(base_pip_deps.values())})

    # Write merged environment file
    print('Writing to:', os.path.abspath('$OUTPUT_FILE'), file=sys.stderr)
    with open('$OUTPUT_FILE', 'w') as f:
        yaml.dump(merged_env, f, default_flow_style=False)
    print('Merge completed successfully', file=sys.stderr)
except Exception as e:
    print(f'Error during Python merge: {str(e)}', file=sys.stderr)
    sys.exit(1)
"



# Check if Python merging succeeded
if [ $? -ne 0 ]; then
echo "Python merging failed. Using simple file concatenation method."
    # Simple concatenation as fallback
    (
        # Get name from base file
        grep "name:" "$BASE_ENV_FILE" | head -n1

        # Get channels from both files (combine unique)
        echo "channels:"
        (grep -A5 "channels:" "$BASE_ENV_FILE" | grep -v "channels:" | grep -v "dependencies:" &&
            grep -A5 "channels:" "$DEV_ENV_FILE" | grep -v "channels:" | grep -v "dependencies:") | sort | uniq | grep -v "^$" | grep -v "^--$"

        # Start dependencies section
        echo "dependencies:"

        # Get conda dependencies from base file
        grep -A100 "dependencies:" "$BASE_ENV_FILE" | grep -v "dependencies:" | grep -v "pip:" | grep -v "  -" | grep "^-" || true

        # Get conda dependencies from dev file
        grep -A100 "dependencies:" "$DEV_ENV_FILE" | grep -v "dependencies:" | grep -v "pip:" | grep -v "  -" | grep "^-" || true

        # Get pip dependencies from both files
        echo "- pip:"
        (grep -A100 "pip:" "$BASE_ENV_FILE" | grep -v "pip:" | grep "  -" &&
            grep -A100 "pip:" "$DEV_ENV_FILE" | grep -v "pip:" | grep "  -") | sort | uniq || true

    ) > "$OUTPUT_FILE"
fi


echo "Created merged environment file: $OUTPUT_FILE"
echo ""



if [ -z "$ENV_MANAGER" ]; then
    echo "Error: No environment manager (conda, micromamba, or mamba) found."
    echo "Please make sure one of these is installed and available in your PATH."
    echo "The merged environment file is still available at: $OUTPUT_FILE"
    exit 1
fi

echo "Using $ENV_MANAGER to create the environment"

if [ "$CREATE_ENV" = true ]; then
  echo "Using $ENV_MANAGER to create the environment"

  # Handle absolute paths for executable if detected
  if [[ "$ENV_MANAGER" == /* ]]; then
      echo "$ENV_MANAGER env create -f $OUTPUT_FILE -y"
      "$ENV_MANAGER" env create -f "$OUTPUT_FILE" -y
  else
      echo "$ENV_MANAGER env create -f $OUTPUT_FILE -y"
      $ENV_MANAGER env create -f "$OUTPUT_FILE" -y
  fi
elif [ "$INSTALL_BASE" = true ]; then
  echo "Installing dependencies to base environment (--install-base mode)"

  # Handle absolute paths for executable if detected
  if [[ "$ENV_MANAGER" == /* ]]; then
      echo "$ENV_MANAGER env update --name base --file $OUTPUT_FILE"
      "$ENV_MANAGER" env update --name base --file "$OUTPUT_FILE"
  else
      $ENV_MANAGER env update --name base --file "$OUTPUT_FILE"
  fi
fi

$ENV_MANAGER list
# Clean up
rm $OUTPUT_FILE
