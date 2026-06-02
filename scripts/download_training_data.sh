#!/usr/bin/env bash
# download_training_data.sh — Sequential lithic mesh downloads at ~1000 KB/s
#
# Downloads additional 3D lithic mesh datasets for classifier training.
# Rate-limited to ~1000 KB/s via curl --limit-rate. Each dataset completes
# before the next starts.
#
# Usage:
#   nohup bash scripts/download_training_data.sh &
#
# Datasets:
#   1. Gahagan Bifaces      (20-30 Caddo bifaces, 69 MB)    — Zenodo ✓
#   2. COADS (more)          (~200 additional projectile points) — Zenodo ✓
#   3. Additional sources    (Figshare/Zenodo aggregates)

set -euo pipefail

RAW_DIR="/data/dibble-training/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

RATE="1000k"  # ~1000 KB/s (1 MB/s)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ──────────────────────────────────────────────────
#  1. Gahagan Bifaces  —  Zenodo
#     DOI: 10.5281/zenodo.3457943
#     20-30 Caddo bifaces from southern Caddo area
#     ZIP: 69 MB
# ──────────────────────────────────────────────────
download_gahagan() {
    local out_dir="$RAW_DIR/gahagan_bifaces"
    local done_marker="$out_dir/.done"
    log "=== 1/4  Gahagan Bifaces (Zenodo 3457943) ==="

    if [ -f "$done_marker" ]; then
        log "  Already downloaded, skipping"
        return 0
    fi

    mkdir -p "$out_dir"

    local url="https://zenodo.org/api/records/3457943/files/aksel-blaise/gahaganmorph-gahaganb1.zip/content"
    local zip_path="$out_dir/gahagan.zip"

    log "  Downloading (69 MB) at $RATE..."
    curl -L --limit-rate "$RATE" -o "$zip_path" "$url" --progress-bar 2>&1 || {
        log "  WARNING: Download failed"
        return 1
    }

    if [ -f "$zip_path" ] && [ -s "$zip_path" ]; then
        log "  Extracting..."
        unzip -q -o "$zip_path" -d "$out_dir" || true
        rm -f "$zip_path"
        local count
        count=$(find "$out_dir" -name '*.ply' -o -name '*.stl' -o -name '*.obj' | wc -l)
        log "  Found $count mesh files"
    fi

    touch "$done_marker"
    log "  Gahagan Bifaces complete"
}

# ──────────────────────────────────────────────────
#  2. COADS — additional records  (Zenodo)
#     Uses the existing download_coads.py script
#     Running with --max to get more than current 398
# ──────────────────────────────────────────────────
download_coads_extra() {
    local done_marker="$RAW_DIR/COADS/.done_coads_extra"
    log "=== 2/4  COADS — additional records ==="

    if [ -f "$done_marker" ]; then
        log "  Already downloaded, skipping"
        return 0
    fi

    local current_count
    current_count=$(find "$RAW_DIR/COADS/ply" -name '*.ply' 2>/dev/null | wc -l)
    log "  Currently have $current_count COADS PLY files"

    # Run the existing downloader with --max 0 (all available)
    # and --convert (GLB → PLY)
    # Rate-limited via the script's own 0.5s delay between records
    log "  Fetching more COADS records (rate-limited via script)..."
    python3 "$PROJECT_ROOT/lithicore/data/training/download_coads.py" \
        --dest "$RAW_DIR/COADS" \
        --max 0 \
        --convert 2>&1 || {
        log "  WARNING: COADS downloader failed (may be API rate-limited)"
    }

    local new_count
    new_count=$(find "$RAW_DIR/COADS/ply" -name '*.ply' 2>/dev/null | wc -l)
    log "  Now have $new_count COADS PLY files (+$((new_count - current_count)))"

    touch "$done_marker"
    log "  COADS extra complete"
}

# ──────────────────────────────────────────────────
#  3. Additional Zenodo sources
#     Checks multiple Zenodo records for downloadable 3D mesh files
# ──────────────────────────────────────────────────
download_zenodo_record() {
    local rec_id="$1"
    local out_dir="$2"
    local desc="$3"
    local done_marker="$out_dir/.done"

    if [ -f "$done_marker" ]; then
        log "  Already downloaded, skipping"
        return 0
    fi

    mkdir -p "$out_dir"
    log "  Fetching Zenodo record $rec_id ($desc)..."

    # Get file listing from Zenodo API
    local api_response
    api_response=$(curl -s -H "User-Agent: Mozilla/5.0" \
        "https://zenodo.org/api/records/$rec_id" 2>/dev/null || echo "{}")

    # Parse file URLs and names
    local file_list
    file_list=$(echo "$api_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    files = data.get('files', [])
    if files:
        for f in files:
            print(f['links']['self'])
            print(f['key'])
    else:
        # Maybe it redirects to a parent or different format
        print('NO_FILES')
except:
    print('PARSE_ERROR')
" 2>/dev/null)

    if [ "$file_list" = "NO_FILES" ] || [ "$file_list" = "PARSE_ERROR" ]; then
        log "  No files available for record $rec_id"
        touch "$done_marker"
        return 0
    fi

    # Download each file in the record
    local download_url=""
    local filename=""
    local file_count=0
    while IFS= read -r line; do
        if [ -z "$download_url" ]; then
            download_url="$line"
        else
            filename="$line"
            file_count=$((file_count + 1))
            # Skip code/supplementary files — only grab mesh archives
            if echo "$filename" | grep -qiE '\.(zip|rar|tar\.gz|7z)$'; then
                log "    Downloading $filename..."
                curl -L --limit-rate "$RATE" -o "$out_dir/$filename" \
                    "$download_url" --progress-bar 2>&1 || {
                    log "    WARNING: Failed to download $filename"
                }
            else
                log "    Skipping non-archive: $filename"
            fi
            download_url=""
            filename=""
        fi
    done <<< "$file_list"

    if [ "$file_count" -eq 0 ]; then
        log "  No downloadable files found"
    fi

    touch "$done_marker"
    log "  Zenodo record $rec_id complete"
}

# ──────────────────────────────────────────────────
#  4. Later Acheulean Handaxes — extra sites
#     Zenodo 14534846 — check if we need NahalZihor or Jaljulia
# ──────────────────────────────────────────────────
check_acheulean() {
    log "=== 4/4  Later Acheulean Handaxes — site check ==="

    # We already have files for all 4 sites in levantine_handaxes/
    # but the Zenodo archives might contain additional meshes
    local lev_dir="$RAW_DIR/levantine_handaxes"
    if [ -d "$lev_dir" ]; then
        local ply_count
        ply_count=$(find "$lev_dir" -name '*.ply' | wc -l)
        log "  Already have $ply_count PLY files across all sites — skipping re-download"
    fi
    log "  Acheulean handaxes check complete"
}

# ── Main ──
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

log "Starting lithic training data downloads at $(date)"
log "Rate limit: $RATE"
log ""

download_gahagan
echo ""
download_coads_extra
echo ""
check_acheulean
echo ""

log ""
log "All downloads complete at $(date)"
log "Summary:"
for d in "$RAW_DIR"/gahagan_bifaces "$RAW_DIR"/COADS; do
    if [ -d "$d" ]; then
        count=$(find "$d" -name '*.ply' -o -name '*.stl' -o -name '*.obj' | wc -l)
        log "  $(basename $d): $count mesh files"
    fi
done
log ""
log "To process new data: python3 lithicore/data/training/download_and_process.py"
log "To retrain:          python3 lithicore/data/training/retrain.py"
