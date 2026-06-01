#!/usr/bin/env bash
set -euo pipefail

CONFIG="pipeline/configs/in_house_caco2.json"

python pipeline/lnp_pipeline.py inspect --config "$CONFIG" --fold 0
python pipeline/lnp_pipeline.py train --config "$CONFIG" --fold 0 --clean "$@"
