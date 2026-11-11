"""Download only data/configuration for the official PARA checkpoint."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "cache")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import snapshot_download

repo = "Thunderbolt215215/ArtiMuse_PARA"
revision = "b44f36e1dd1fae102b4c2587ebf95e324ea57bc4"
print(f"Downloading {repo} at {revision}", flush=True)
snapshot_download(repo, revision=revision, local_dir=ROOT / "checkpoint_para",
                  allow_patterns=["*.safetensors", "*.json", "merges.txt", "*.model"],
                  token=False, max_workers=4)
(ROOT / "para_revision.json").write_text(json.dumps(dict(repo=repo, revision=revision), indent=2))
print("Download complete", flush=True)
