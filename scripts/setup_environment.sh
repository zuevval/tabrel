#!/bin/bash

micromamba create -f environment.yml

eval "$(micromamba shell hook --shell bash)"
micromamba activate tabrel

# Install pre-commit hooks
pre-commit install
