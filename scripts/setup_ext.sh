#!/bin/bash
set -e

# ============================================================
# Utility functions
# ============================================================

# Clone a repository if it doesn't already exist
clone_repo() {
    local repo_url=$1
    local target_dir=$2

    if [ ! -d "$target_dir/.git" ]; then
        echo "→ Cloning $repo_url to $target_dir ..."
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir"
    else
        echo "✔ Repository already exists at $target_dir, skipping clone."
    fi
}

# Check and install local editable Python packages
install_modules() {
    local modules=("$@")

    echo "📦 Installing local editable modules..."
    for module in "${modules[@]}"; do
        if [ -d "$module" ]; then
            echo "→ Installing $module ..."
            pip install -e "$module"
        else
            echo "⚠ Skipped $module (directory not found)"
        fi
    done
}

# ============================================================
# Main script
# ============================================================

echo "🧩 Checking and cloning required repositories..."

clone_repo git@github.com:Renforce-Dynamics/assetslib.git ./data/assets/assetslib
clone_repo git@github.com:Renforce-Dynamics/robotlib.git ./source/robotlib
clone_repo git@github.com:Renforce-Dynamics/IsaacNPC.git ./source/third_party/IsaacNPC
clone_repo git@github.com:Renforce-Dynamics/fsmLab.git ./source/third_party/fsmLab

modules=(
    "./source/rsl_rl"
    "./source/soccerLab"
    "./source/soccerTask"
    "./source/robotlib"
    "./source/third_party/beyondMimic"
    "./source/third_party/locomotion_rl_lab"
    "./source/third_party/sim2simlib"
    "./source/third_party/IsaacNPC"
    "./source/third_party/fsmLab"
)

install_modules "${modules[@]}"

echo "✅ All done!"
