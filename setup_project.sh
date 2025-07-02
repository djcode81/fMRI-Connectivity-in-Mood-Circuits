#!/bin/bash

echo "Setting up fMRI mood circuits project..."

mkdir -p data/{raw,derivatives}
mkdir -p scripts
mkdir -p results
mkdir -p docs

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Initializing git repository..."
git init
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo "data/raw/*" >> .gitignore
echo "data/derivatives/*" >> .gitignore
echo "!data/raw/.gitkeep" >> .gitignore
echo "!data/derivatives/.gitkeep" >> .gitignore

touch data/raw/.gitkeep
touch data/derivatives/.gitkeep

echo "Project structure created in $(pwd)"
ls -la
