#!/usr/bin/env bash
# download_additional_training_data.sh — Sequential download of new lithic datasets
#
# Downloads additional 3D lithic mesh datasets identified by Deep Research
# for expanding the Dibble classifier training corpus. Targets under-represented
# classes: Retouched Flake, Unmodified Flake, Unmodified Cobble.
#
# Rate-limited to ~1000 KB/s via curl --limit-rate.
# Each dataset completes before the next starts. Designed to run overnight.
#
# Usage:
#   nohup bash scripts/download_additional_training_data.sh &
#   tail -f /tmp/additional_training_data.log
#
# Datasets:
#   1. 3D-EdgeAngle validation corpus       3 STL files, ~600 MB   — Zenodo ✓
#   2. UMN DRUM experimental lithics        30 OBJ files, ~500 MB  — DRUM (needs verification)
#   3. Brazil unifacial tools               67 PLY/OBJ,  ~500 MB  — needs manual sourcing
#   4. Hammerstone baseline cobbles         variable               — Royal Society
#   5. Clast classification (cobbles)       variable               — Earth Science

set -euo pipefail

RAW_DIR="/data/dibble-training/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"
RATE="1000k"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
done_marker() { local d="$1"; echo "$d"; touch "$d/.done"; }

# ──────────────────────────────────────────────────
#  1. 3D-EdgeAngle Validation Corpus
#     Zenodo 10.5281/zenodo.7326242
#     3 high-resolution STL files for edge-angle algorithm validation
#     Contains: EAP-flake (minimally retouched), BU-072 (Keilmesser),
#               WEM-60 (archaeological tool)
#     Maps to: Retouched Flake
# ──────────────────────────────────────────────────
download_edgeangle() {
    local out_dir="$RAW_DIR/edgeangle_validation"
    local done="$out_dir/.done"
    log "=== 1/5  3D-EdgeAngle Validation Corpus ==="
    if [ -f "$done" ]; then log "  Already downloaded, skipping"; return 0; fi
    mkdir -p "$out_dir"

    # Zenodo file download URLs (confirmed working)
    local files=(
        "https://zenodo.org/api/records/7326242/files/EAP-flake.stl/content"
        "https://zenodo.org/api/records/7326242/files/BU-072.stl/content"
        "https://zenodo.org/api/records/7326242/files/WEM-60.stl/content"
    )
    local names=("EAP-flake.stl" "BU-072.stl" "WEM-60.stl")
    local sizes=("142 MB" "140 MB" "321 MB")

    for i in "${!files[@]}"; do
        local target="$out_dir/${names[$i]}"
        if [ -f "$target" ]; then
            log "  Already have: ${names[$i]}"
            continue
        fi
        log "  Downloading ${names[$i]} (${sizes[$i]})..."
        curl -L --limit-rate "$RATE" -o "$target" "${files[$i]}" --progress-bar 2>&1 || {
            log "  WARNING: Failed to download ${names[$i]}"
        }
    done

    # Verify
    local count
    count=$(find "$out_dir" -name '*.stl' | wc -l)
    log "  Downloaded $count STL files"
    touch "$done"
    log "  3D-EdgeAngle complete"
}

# ──────────────────────────────────────────────────
#  2. UMN DRUM Experimental Lithics
#     DOI: 10.13020/D6T88N
#     30 OBJ files of experimentally-produced flakes and cores
#     Pristine ground-truth unmodified flakes from Keokuk chert
#     Maps to: Unmodified Flake, Experimental Core
#
#     NOTE: Download URL may require authentication.
#     If this fails, visit the page manually:
#     https://conservancy.umn.edu/items/42498dea-f904-43a5-8cdc-9aab61a82dd9
# ──────────────────────────────────────────────────
download_umn_drum() {
    local out_dir="$RAW_DIR/umn_drum_experimental"
    local done="$out_dir/.done"
    log "=== 2/5  UMN DRUM Experimental Lithics ==="
    if [ -f "$done" ]; then log "  Already downloaded, skipping"; return 0; fi
    mkdir -p "$out_dir"

    # Try multiple possible download URLs
    local urls=(
        "https://conservancy.umn.edu/bitstreams/da03b17b-04be-4a70-a311-5f9bc320f9e1/download"
        "https://conservancy.umn.edu/bitstream/handle/11299/180304/DRUM_Lithic_Models.zip"
    )

    local downloaded=false
    for url in "${urls[@]}"; do
        log "  Trying: $url"
        if curl -sI --max-time 10 "$url" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -q '200\|302\|206'; then
            log "  Downloading..."
            curl -L --limit-rate "$RATE" -o "$out_dir/drum_lithics.zip" "$url" --progress-bar 2>&1 && {
                log "  Extracting..."
                unzip -q -o "$out_dir/drum_lithics.zip" -d "$out_dir" 2>/dev/null || true
                rm -f "$out_dir/drum_lithics.zip"
                local count
                count=$(find "$out_dir" -name '*.obj' -o -name '*.ply' -o -name '*.stl' | wc -l)
                log "  Extracted $count mesh files"
                downloaded=true
                break
            }
        fi
    done

    if [ "$downloaded" = false ]; then
        log "  WARNING: Could not download UMN DRUM dataset."
        log "  The repository may require authentication."
        log "  Manual URL: https://conservancy.umn.edu/items/42498dea-f904-43a5-8cdc-9aab61a82dd9"
    fi

    touch "$done"
    log "  UMN DRUM complete"
}

# ──────────────────────────────────────────────────
#  3. Holocene Unifacial Tools — Brazil
#     PLOS One: 10.1371/journal.pone.0315746
#     67 unifacially shaped tools from Central Brazil
#     Maps to: Retouched Flake (unifacial tools from South America)
#
#     NOTE: The Figshare supplement (28126957) only contains the analysis
#     Excel file. The actual 3D meshes may require emailing the authors or
#     checking MorphoSource. The PLOS One article is open access.
# ──────────────────────────────────────────────────
download_brazil() {
    local out_dir="$RAW_DIR/brazil_unifacial_tools"
    local done="$out_dir/.done"
    log "=== 3/5  Holocene Unifacial Tools (Brazil) ==="
    if [ -f "$done" ]; then log "  Already downloaded, skipping"; return 0; fi
    mkdir -p "$out_dir"

    log "  NOTE: 3D meshes are not publicly archived on Figshare/Zenodo."
    log "  The PLOS One article's Figshare supplement (28126957) contains"
    log "  only the analysis Excel file, not the 3D meshes."
    log ""
    log "  To source: email corresponding author González-Varas et al."
    log "  or check MorphoSource for the 67 PLY/OBJ files."
    log "  Article: https://doi.org/10.1371/journal.pone.0315746"

    touch "$done"
    log "  Brazil dataset — requires manual author contact"
}

# ──────────────────────────────────────────────────
#  4. Experimental Hammerstone Baselines
#     Royal Society: rsif.2021.0576
#     Pre-use cobble scans from primate percussive tool studies
#     Maps to: Unmodified Cobble
#
#     NOTE: Requires sourcing from the paper's supplementary data.
#     https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0576
# ──────────────────────────────────────────────────
download_hammerstone() {
    local out_dir="$RAW_DIR/hammerstone_cobbles"
    local done="$out_dir/.done"
    log "=== 4/5  Experimental Hammerstone Cobbles ==="
    if [ -f "$done" ]; then log "  Already downloaded, skipping"; return 0; fi
    mkdir -p "$out_dir"

    log "  NOTE: Hammerstone STL files need manual sourcing."
    log "  Check the Royal Society paper for supplementary data:"
    log "  https://doi.org/10.1098/rsif.2021.0576"
    log "  Search Zenodo for 'hammerstone 3D scan STL'"

    touch "$done"
    log "  Hammerstone dataset — requires manual sourcing"
}

# ──────────────────────────────────────────────────
#  5. Earth Science Clast Classification
#     Plos One: PMC9365149
#     3D point clouds of river cobbles for ML clast classification
#     Maps to: Unmodified Cobble
#
#     NOTE: Requires sourcing from Earth Science repositories.
#     Search for "fluvial clast 3D point cloud" on Zenodo/Figshare.
# ──────────────────────────────────────────────────
download_clasts() {
    local out_dir="$RAW_DIR/earth_science_clasts"
    local done="$out_dir/.done"
    log "=== 5/5  Earth Science Clast Classification ==="
    if [ -f "$done" ]; then log "  Already downloaded, skipping"; return 0; fi
    mkdir -p "$out_dir"

    log "  NOTE: Earth science clast datasets need manual sourcing."
    log "  Search Zenodo for: fluvial clast 3D point cloud"
    log "  Reference paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9365149/"

    touch "$done"
    log "  Clast dataset — requires manual sourcing"
}

# ── Main ──
log "Starting additional lithic data downloads at $(date)"
log "Rate limit: $RATE"
log ""

download_edgeangle
echo ""
download_umn_drum
echo ""
download_brazil
echo ""
download_hammerstone
echo ""
download_clasts
echo ""

log ""
log "All downloads attempted at $(date)"
log "Summary:"
for d in "$RAW_DIR"/edgeangle_validation "$RAW_DIR"/umn_drum_experimental \
          "$RAW_DIR"/brazil_unifacial_tools "$RAW_DIR"/hammerstone_cobbles \
          "$RAW_DIR"/earth_science_clasts; do
    if [ -d "$d" ]; then
        count=$(find "$d" -name '*.stl' -o -name '*.obj' -o -name '*.ply' | wc -l)
        log "  $(basename $d): $count mesh files"
    fi
done
log ""
log "Datasets needing manual follow-up:"
log "  1. Brazil unifacial tools — email authors / check MorphoSource"
log "  2. Hammerstone cobbles — check Royal Society supplementary data"
log "  3. Earth science clasts — search Zenodo for fluvial 3D point clouds"
log "  4. UMN DRUM — try visiting the page manually for download"
