#!/usr/bin/env bash
# download_training_data.sh — Sequential lithic mesh downloads at ~1000 KB/s
#
# Downloads additional 3D lithic mesh datasets for classifier training.
# Rate-limited to ~1000 KB/s via curl --limit-rate. Each dataset completes
# before the next starts. Designed to run overnight.
#
# Usage:
#   nohup bash scripts/download_training_data.sh &
#
# Datasets (Step 1):
#   1. Nubian Levallois Database  (~200 meshes, OSF)
#   2. Figshare Aggregate         (~100 meshes, Figshare/Zenodo)
#   3. Selden Paleoindian         (50-100 meshes, Zenodo)

set -euo pipefail

RAW_DIR="/data/dibble-training/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

RATE="1000k"  # ~1000 KB/s

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ──────────────────────────────────────────────────
#  1. Nubian Levallois Database  —  OSF
#     DOI: 10.17605/OSF.IO/SJ8ZV
#     ~200 PLY meshes, labelled Levallois cores
# ──────────────────────────────────────────────────
download_nubian() {
    local out_dir="$RAW_DIR/nubian_levallois"
    mkdir -p "$out_dir"
    log "=== 1/3  Nubian Levallois Database ==="
    log "  Target: $out_dir"

    # OSF download URL for the project's main file storage
    local osf_url="https://osf.io/download/sj8zv/"
    local zip_path="$out_dir/nubian_levallois.zip"

    if [ -f "$out_dir/.done" ]; then
        log "  Already downloaded, skipping"
        return 0
    fi

    log "  Downloading from OSF (rate-limited to $RATE)..."
    curl -L --limit-rate "$RATE" -o "$zip_path" "$osf_url" || {
        log "  WARNING: OSF download failed. The dataset may require manual download."
        log "  URL: https://osf.io/sj8zv/"
        return 1
    }

    if [ -f "$zip_path" ] && [ -s "$zip_path" ]; then
        log "  Extracting..."
        unzip -q -o "$zip_path" -d "$out_dir" || true
        rm -f "$zip_path"
        log "  Extracted to $out_dir"
        # Recursively find PLY files
        local count
        count=$(find "$out_dir" -name '*.ply' | wc -l)
        log "  Found $count PLY files"
    fi

    touch "$out_dir/.done"
    log "  Nubian Levallois complete"
}

# ──────────────────────────────────────────────────
#  2. Figshare Aggregate  —  multiple DOIs
#     ~100+ meshes across several figshare collections
# ──────────────────────────────────────────────────
download_figshare() {
    local out_dir="$RAW_DIR/figshare_aggregate"
    mkdir -p "$out_dir"
    log "=== 2/3  Figshare Aggregate ==="

    if [ -f "$out_dir/.done" ]; then
        log "  Already downloaded, skipping"
        return 0
    fi

    # Figshare collections with lithic 3D meshes
    local figshare_urls=(
        "https://figshare.com/ndownloader/articles/22635828"   # Handaxes and cleavers
        "https://figshare.com/ndownloader/articles/19186143"   # Biśnik Cave
    )

    local i=0
    for url in "${figshare_urls[@]}"; do
        i=$((i+1))
        local zip_path="$out_dir/figshare_${i}.zip"
        log "  Downloading figshare collection $i..."
        curl -L --limit-rate "$RATE" -o "$zip_path" "$url" || {
            log "  WARNING: Figshare download $i failed"
            continue
        }
        if [ -f "$zip_path" ] && [ -s "$zip_path" ]; then
            log "  Extracting..."
            unzip -q -o "$zip_path" -d "$out_dir/collection_${i}" || true
            rm -f "$zip_path"
            local count
            count=$(find "$out_dir/collection_${i}" -name '*.ply' -o -name '*.obj' -o -name '*.stl' | wc -l)
            log "  Found $count mesh files in collection $i"
        fi
    done

    touch "$out_dir/.done"
    log "  Figshare aggregate complete"
}

# ──────────────────────────────────────────────────
#  3. Selden Paleoindian Collections  —  Zenodo
#     Clovis points, Paleoindian bifaces from Gault/Blackwater Draw
# ──────────────────────────────────────────────────
download_selden() {
    local out_dir="$RAW_DIR/selden_paleoindian"
    mkdir -p "$out_dir"
    log "=== 3/3  Selden Paleoindian Collections ==="

    if [ -f "$out_dir/.done" ]; then
        log "  Already downloaded, skipping"
        return 0
    fi

    # Zenodo records for Selden collections
    local zenodo_ids=(
        "3457943"   # Gahagan Bifaces (Caddo)
    )

    local i=0
    for rec_id in "${zenodo_ids[@]}"; do
        i=$((i+1))

        # Get file list from Zenodo API (unauthenticated, max 25 per page)
        log "  Fetching Zenodo record $rec_id..."
        local files_json
        files_json=$(curl -s "https://zenodo.org/api/records/$rec_id" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for f in data.get('files', []):
        print(f['links']['self'])
        print(f['key'])
except: pass
" 2>/dev/null || true)

        if [ -z "$files_json" ]; then
            log "  WARNING: No files found for Zenodo record $rec_id"
            continue
        fi

        # Download each file
        local download_url=""
        local filename=""
        while IFS= read -r line; do
            if [ -z "$download_url" ]; then
                download_url="$line"
            else
                filename="$line"
                log "    Downloading $filename..."
                curl -L --limit-rate "$RATE" -o "$out_dir/$filename" "$download_url" || {
                    log "    WARNING: Failed to download $filename"
                }
                download_url=""
                filename=""
            fi
        done <<< "$files_json"
    done

    touch "$out_dir/.done"
    log "  Selden Paleoindian complete"
}

# ── Main ──
log "Starting lithic training data downloads at $(date)"
log "Rate limit: $RATE"
log ""

download_nubian
echo ""
download_figshare
echo ""
download_selden

log ""
log "All downloads complete at $(date)"
log "Summary:"
for d in "$RAW_DIR"/nubian_levallois "$RAW_DIR"/figshare_aggregate "$RAW_DIR"/selden_paleoindian; do
    if [ -d "$d" ]; then
        count=$(find "$d" -type f | wc -l)
        log "  $(basename $d): $count files"
    fi
done
