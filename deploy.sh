#!/usr/bin/env bash

# Packages and uploads a new version to PyPi.

set -e

# Activate the venv.
source venv/bin/activate

# Clean out old build artifacts.
rm -rf dist/ ./*.egg-info/ build/
# Create the package.
python3 setup.py sdist bdist
# Upload.
python3 -m twine upload dist/*.linux-x86_64.tar.gz
