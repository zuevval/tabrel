#!/bin/bash

micromamba create -f environment.yml
micromamba activate tabrel

# Install pre-commit hooks
pre-commit install
