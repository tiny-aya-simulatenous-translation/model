#!/bin/bash
# One-time GCP bootstrap for TinyAya Stage 2 TPU training.
# Run from your local workstation (not from a TPU VM) after `gcloud auth login`.
#
# What this does:
#   1. Enables required APIs.
#   2. Creates the GCS bucket for checkpoints.
#   3. Reads HF_TOKEN / WANDB_API_KEY from the repo-root .env and pushes them
#      into Secret Manager.
#   4. Grants the TPU service account read access to those secrets and write
#      access to the bucket.
#
# Idempotent: safe to re-run.

set -euo pipefail

# Resolve repo root (this script lives in <repo>/simultaneous-translation/scripts/tpu/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Create it with HF_TOKEN and (optionally) WANDB_API_KEY."
    exit 1
fi

# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$ENV_FILE"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
REGION="${REGION:-us-central2}"
BUCKET="${BUCKET:-tinyaya-stage2-tpu}"
SECRET_HF="${SECRET_HF:-hf-token}"
SECRET_WANDB="${SECRET_WANDB:-wandb-api-key}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN missing in $ENV_FILE"
    exit 1
fi

echo "==> using project: $PROJECT_ID  region: $REGION  bucket: gs://$BUCKET"
gcloud config set project "$PROJECT_ID"

# ---- 1. APIs ----
echo "==> enabling APIs"
gcloud services enable \
    tpu.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    compute.googleapis.com

# ---- 2. bucket ----
echo "==> creating bucket gs://$BUCKET (no-op if exists)"
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://$BUCKET" \
        --location="$REGION" \
        --uniform-bucket-level-access
else
    echo "    bucket already exists"
fi

# ---- 3. secrets ----
push_secret() {
    local name="$1" value="$2"
    if [ -z "$value" ]; then
        echo "    skipping secret $name (empty value)"
        return
    fi
    if gcloud secrets describe "$name" >/dev/null 2>&1; then
        echo -n "$value" | gcloud secrets versions add "$name" --data-file=-
        echo "    rotated secret $name (new version)"
    else
        echo -n "$value" | gcloud secrets create "$name" --data-file=- --replication-policy=automatic
        echo "    created secret $name"
    fi
}

# Strip surrounding quotes that may exist in .env (e.g. HF_TOKEN="...").
HF_TOKEN_CLEAN="${HF_TOKEN%\"}"; HF_TOKEN_CLEAN="${HF_TOKEN_CLEAN#\"}"
WANDB_API_KEY_CLEAN="${WANDB_API_KEY:-}"
WANDB_API_KEY_CLEAN="${WANDB_API_KEY_CLEAN%\"}"; WANDB_API_KEY_CLEAN="${WANDB_API_KEY_CLEAN#\"}"

echo "==> seeding Secret Manager"
push_secret "$SECRET_HF" "$HF_TOKEN_CLEAN"
push_secret "$SECRET_WANDB" "$WANDB_API_KEY_CLEAN"

# ---- 4. IAM ----
# On a fresh project the TPU service agent is not auto-created just by
# enabling the API; we have to ask for it explicitly. This call is idempotent.
echo "==> ensuring TPU service identity exists"
gcloud beta services identity create --service=tpu.googleapis.com --project="$PROJECT_ID" >/dev/null

PROJECT_NUM="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
TPU_SA="service-${PROJECT_NUM}@cloud-tpu.iam.gserviceaccount.com"
# The TPU VM itself runs as the project's default Compute Engine service
# account, which is what the startup script uses for `gcloud secrets ...`
# and `gcloud storage ...` calls.
VM_SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"

echo "==> granting IAM to TPU service identity: $TPU_SA"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$TPU_SA" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None >/dev/null

gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member="serviceAccount:$TPU_SA" \
    --role="roles/storage.objectAdmin" >/dev/null

echo "==> granting IAM to default Compute Engine SA (used by the TPU VM): $VM_SA"
for sec in "$SECRET_HF" "$SECRET_WANDB"; do
    if gcloud secrets describe "$sec" >/dev/null 2>&1; then
        gcloud secrets add-iam-policy-binding "$sec" \
            --project="$PROJECT_ID" \
            --member="serviceAccount:$VM_SA" \
            --role="roles/secretmanager.secretAccessor" >/dev/null
    fi
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member="serviceAccount:$VM_SA" \
    --role="roles/storage.objectAdmin" >/dev/null

echo "==> done. Next: bash scripts/tpu/launch_qr.sh"
