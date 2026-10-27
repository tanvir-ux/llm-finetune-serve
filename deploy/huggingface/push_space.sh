#!/usr/bin/env bash
# Publish the Docker demo as a Hugging Face Space. Requires a write token in HF_TOKEN.
# NOTE: not executed for this repository (no Hugging Face token was available at build time).
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"
USER_NAME="${HF_USER:-tanvir-ux}"
SPACE="${USER_NAME}/llm-finetune-serve"
WORK="$(mktemp -d)"
git clone --depth 1 . "$WORK/src"
rm -rf "$WORK/src/.git" "$WORK/src/docs" "$WORK/src/tests" "$WORK/src/artifacts"
cp deploy/huggingface/README.md "$WORK/src/README.md"   # Space metadata replaces the repo README
pip install -q huggingface_hub
python - "$SPACE" "$WORK/src" <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(sys.argv[1], repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=sys.argv[1], repo_type="space", folder_path=sys.argv[2])
print("https://huggingface.co/spaces/" + sys.argv[1])
PY
