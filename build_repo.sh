#!/bin/bash

# go to location of this script
# cd $(dirname $0)
# remove browser-use folder if it exists
# rm -rf browser-use

# Simple test script for browser-use Gemini branch
git clone git@github.com:agi-inc/browser-use.git
cd browser-use
# git checkout models/gemini
# git checkout hot-fix/default-not-permitted
# checkout 1a5a4bd557a190dbc6a77985c46db383779ae4bf
echo "Checking out branch"
git checkout MultionDOMParser


# Setup environment and build
# echo "Setting up environment and building"
cd ..
uv venv --python=3.11
source .venv/bin/activate
cd browser-use
uv pip install build 
python -m build

echo "Installing browser-use"
uv pip install dist/browser_use-*.whl
uv pip install langchain_google_genai

echo "Installing playwright"
playwright install


# Run test
# copy test.py to browser-use
#echo "Copying test.py to browser-use"
cd ..
source .venv/bin/activate
#cp test.py browser-use/test.py
#cd browser-use



# deactivate

# activate
# cd browser-use
# source .venv/bin/activate
# cd ..
# python test.py


# c
# i browser-use

# a .venv