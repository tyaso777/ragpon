# FastAPI Local Models

This directory is reserved for local Hugging Face model files used by the
FastAPI app. Model files are intentionally not committed to Git.

After cloning this repository, prepare the embedding model under this directory:

```text
ragpon/apps/fastapi/models/
└── models--cl-nagoya--ruri-v3-310m/
    └── snapshots/
        └── <revision>/
```

Then set `MODELS.CL_NAGOYA_RURI_V3_MODEL_PATH` in the FastAPI config to the
snapshot directory, for example:

```yaml
MODELS:
  CL_NAGOYA_RURI_V3_MODEL_PATH: "/app/ragpon/apps/fastapi/models/models--cl-nagoya--ruri-v3-310m/snapshots/18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"
```

Other local model directories can also be placed here as needed. Keep generated
model files out of Git.
