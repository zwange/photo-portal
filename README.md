# Photo Portal

A local gallery and read-only SQL workspace for browsing a personal photo
collection. It ranks thumbnails with ArtiMuse_PARA and uses a separate
hierarchical classifier for categories, descriptions, and subject and scene tags.
The browser can open an original file or reveal it in Windows File Explorer.

## Setup on Windows

Use Python 3.12 (regular Python or an Anaconda environment) and Git. The gallery
itself needs no GPU, Ollama server, or Node.js.

```powershell
git clone https://github.com/zwange/photo-portal.git
Set-Location photo-portal
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config.example.json config.json
```

Edit `config.json` to point `photo_roots` at one or more local image directories
and `database` at the SQLite catalog. Paths may be absolute or relative to the
config file; use forward slashes in JSON. Roots must not overlap. Use the optional
`hidden_folders` list of absolute paths to hide folders and their descendants
from the gallery and media routes. This is a display preference: explicit SQL
queries can still read their catalog records.

If you already have a compatible catalog, point the config at it and run
`python server.py` or `python launch.py` (which opens the browser). Otherwise,
start with the local indexing and model steps below. The portal binds to
loopback, normally [port 8787](http://127.0.0.1:8787/). If that port is occupied,
check the console or the ignored `server.json` for the actual URL.

## Build a catalog from photos

1. Run `python catalog.py`. It indexes supported still images (JPEG, PNG, WebP,
   TIFF and BMP), records EXIF when available, and writes JPEG thumbnails with
   a longest edge of 1024 pixels under each root's `.photo-portal-thumbs` folder.
   Originals are read-only. Rerunning indexes new images and retries failed
   thumbnails; it is not a file-change or deletion synchronizer.
2. Install [Ollama](https://ollama.com/download), make `ollama` available on
   `PATH`, and run `ollama pull qwen3.5:4b`. Then run
   `python classify_hierarchy.py`. The classifier starts a local Ollama worker
   on port 11435, uses the existing taxonomy (including Vehicles → Aircraft),
   and resumes without redoing successful labels.
3. Score the catalog with PARA using the steps below. Gallery cards require
   both a successful classification and a score.
4. Optionally install `requirements-visual.txt` and run
   `python visual_full.py` to build visual version groups. The gallery uses the
   latest completed full run. Exact-copy removal is available without this step.

## ArtiMuse_PARA scoring

Checkpoint: [Hugging Face model card](https://huggingface.co/Thunderbolt215215/ArtiMuse_PARA)
· [pinned Hugging Face files](https://huggingface.co/Thunderbolt215215/ArtiMuse_PARA/tree/b44f36e1dd1fae102b4c2587ebf95e324ea57bc4)
· [ModelScope mirror](https://modelscope.cn/models/thunderbolt/ArtiMuse_PARA).
The [official ArtiMuse source](https://github.com/thunderbolt215/ArtiMuse)
is a separate checkout. This repository contains no weights or third-party
model source.

Scoring requires an NVIDIA GPU with BF16 support. The tested Windows setup
uses an RTX 4090, PyTorch 2.10 / torchvision 0.25 with CUDA 13.0, and batch 4.
The checkpoint download is about 16 GB. Choose a compatible PyTorch build
from the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v2100).
For the tested configuration, run from the repository root:

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-scoring.txt
git clone https://github.com/thunderbolt215/ArtiMuse.git vendor/ArtiMuse
git -C vendor/ArtiMuse checkout 750d980b6b7e9d99da60a302dcdbcab14e01003f
python download_para.py
python score_para_catalog.py
```

The downloader pins the checkpoint revision and stores the weights in ignored
`checkpoint_para/`. Scoring uses the catalog thumbnails and records numeric
scores only. Progress and resumable results are under `para_full_run/`; rerun
the same command to resume. Once every eligible thumbnail has a valid score,
one transaction publishes the scores to `photos.artimuse_score` and records
the model revision in `artimuse_runs`. The staging database retains the previous
scores for recovery. For a fresh full pass after adding photos, archive the
finished `para_full_run/` folder before rerunning.

## Browse and query

- Percentiles measure the share of visible scored photos at or below a given
  PARA score. They stay fixed when changing gallery filters. Raw scores are
  available in photo details and CSV.
- Filter by category, subcategory, camera, exposure, and subject or scene tags.
  Explore mode mixes ranked and random selections.
- The optional shoot cap is shared across all categories. One shoot is a folder
  immediately below a photo root and all its descendants. The photos-per-category
  setting is also a maximum.
- Gallery filters apply before the best eligible version is chosen. SQL table
  results preserve the original rows; switching to SQL Gallery shows the visible
  photos represented by those rows.

```sql
SELECT id, file_name, artimuse_score,
       ROUND(artimuse_percentile(artimuse_score), 2) AS percentile
FROM photos_hierarchical
WHERE classification_status = 'ok' AND artimuse_status = 'ok'
  AND primary_category = 'vehicles' AND subcategory = 'aircraft'
  AND portal_visible(folder_path)
ORDER BY artimuse_score DESC
LIMIT 50;
```

`artimuse_percentile()` and `portal_visible()` are provided in the portal's
SQL workspace. The portal opens the photo catalog read-only; indexing,
classification, scoring, and optional grouping are separate local write jobs.

Run the source smoke test with `python -m unittest discover -s tests`. It uses
synthetic images and a temporary catalog, so it does not need checkpoints or
access to a personal collection.
