# Finding Dibble-Ready Lithic Photogrammetry Benchmarks

## Executive Summary

- **No True Paired Benchmark Found**: I found no publicly indexed lithic dataset that clearly provides both source multi-angle photographs and an independent CT, structured-light, or laser scan of the same artifacts at useful scale -> treat the current public landscape as proxy-only unless you can negotiate photo access from scan-only projects.
- **Best Photo Input Candidate**: The UCL/Figshare Maritime Academy Giant Handaxe records provide **662 RAW DNG photographs** for one handaxe, split across four zip files, and an associated Sketchfab model exists -> use it as a COLMAP smoke test, not as independent geometric ground truth ([executive_summary[0]] [45], [executive_summary[1]] [47], [executive_summary[2]] [48]).
- **Best Scan-Only Ground Truth Pool**: The Open Aurignacian Project publishes **2,016** stone-tool 3D models from four Italian Early Upper Paleolithic sequences, made with Artec Space Spider, Artec Micro, and micro-CT -> excellent ground-truth meshes, but you would need new photographs of the same artifacts or permission from custodians to make them Dibble benchmarks ([executive_summary[3]] [89]).
- **Best Multi-Protocol Photogrammetry Proxy**: Sorrentino et al. document **4 ground-stone replicas**, **32 photogrammetric models**, and two capture strategies with **100-120** or **150-190** photographs per stage -> useful for COLMAP protocol stress testing, but still photogrammetry-only rather than independent scan validation ([executive_summary[4]] [83]).
- **Best Experimental Mesh-Only Proxy**: The University of Minnesota DRUM dataset provides OBJ/MTL/JPG textured 3D models from both expedient and refined photogrammetry protocols -> useful for downstream Dibble mesh-analysis tests, but not for image-to-mesh validation because the source photos are not included ([executive_summary[5]] [28]).
- **Synthetic Gap**: I found virtual-knapping and ML work that programmatically generates lithic geometry, but no public multi-view rendered lithic image dataset designed as a photogrammetry benchmark -> if Dibble needs synthetic validation, generate renders yourself from open PLY meshes and publish camera poses, lighting, and ground-truth meshes ([executive_summary[6]] [50]).
- **Nubian Levallois OSF Reality Check**: The OSF nodes `sj8zv` and `xz7cb` exist, but their public `osfstorage` API roots returned **0** visible files in the endpoints inspected -> do not assume the Nubian OSF project contains bulk photos or meshes without checking linked components/providers ([executive_summary[7]] [40], [executive_summary[8]] [41]).
- **OSF Download Strategy**: Use `osfclient` for simple public project cloning and the OSF API for robust nested-folder downloads; the API exposes pagination, rate limits, and file `links.download` URLs -> build retries, skip-existing logic, and component traversal into any Dibble ingestion script ([executive_summary[9]] [73], [executive_summary[10]] [72]).

## Ranked Dataset Table

| Rank | Suitability for Dibble | Dataset and citation | Platform | Artifacts / models | Multi-angle photos and capture protocol | 3D scan / mesh method and formats | License / access | Limitations |
|---:|---|---|---|---:|---|---|---|---|
| 0 | **Confirmed paired photo + independent scan** | **None found in public sources searched as of 2026-05-29** | N/A | N/A | N/A | N/A | N/A | The central gap is not 3D lithic data; it is paired source-photo retention plus independent scan ground truth. |
| 1 | **Best photo-input proxy** | Matthew Pope and Letty Ingrey, "Zip file containing RAW format image files used to capture the Giant Handaxe" records, including DOI **10.5522/04/23591925** for files 001-160 and DOI **10.5522/04/22957319** for files 161-336 ([ranked_dataset_table[0]] [45], [ranked_dataset_table[1]] [47], [ranked_dataset_table[2]] [69], [ranked_dataset_table[3]] [71]) | UCL Research Data Repository / Figshare; associated model on Sketchfab | 1 handaxe | **662** RAW DNG images: files 001-160, 161-336, 337-551, and 552-662. The records describe close-range photogrammetry with overlapping digital photographs processed to produce 3D geometry. | Associated 3D model available on Sketchfab ([ranked_dataset_table[4]] [48]); independent CT/structured-light scan not documented. Image format: DNG. Model download format not verified from the extracted page. | Image records state CC0 in extracted UCL/Figshare metadata. Sketchfab model license should be checked directly before redistribution. | Excellent for COLMAP pipeline input, but only **one artifact** and the model is not independent ground truth. Four separate zip records complicate ingestion. |
| 2 | **Photogrammetry-only proxy with strong protocol detail** | Sorrentino, Menna, Remondino, Paggi, Longo, Borghi, Re, and Lo Giudice, 2023, "Close-range photogrammetry reveals morphometric changes on replicative ground stones," DOI **10.1371/journal.pone.0289807** ([ranked_dataset_table[5]] [83]); data DOI **10.5281/zenodo.8196541** | PLOS ONE article; Zenodo data | 4 ground-stone tool replicas; 32 models reported | Nikon D750 with AF-S Micro NIKKOR 60 mm; NEF raw images. Ad hoc setup: table rotated at 10 degree intervals, three camera heights, **100-120** images per acquisition. Literature setup: rotating plate, 15 degree intervals, flipping object, **150-190** photos per stage. | SfM/MVS photogrammetry processed in Agisoft Metashape; exported OBJ. No independent CT or structured-light scan reported. | PLOS article is open access; Zenodo dataset linked by DOI. | Strong for protocol benchmarking and repeatability, but not knapped flint; the ad hoc setup involved perforating stones, and the literature-based setup missed the pin side. Photogrammetry scale/orientation errors remain part of the method being tested. |
| 3 | **Photo/RTI documentation proxy, not full 3D ground truth** | Looten, Gravina, Muth, Villaeys, and Bordes, 2025, "Towards a more robust representation of lithic industries in archaeology," DOI **10.5281/zenodo.15411558** ([ranked_dataset_table[6]] [105]; article at [ranked_dataset_table[7]] [29]) | Zenodo / Peer Community Journal | Exact count not clearly stated in the Zenodo extraction; the article discusses an experimental handaxe and an Acheulean handaxe from Cagny l'Epinette | Supplementary RTI data are available; the article notes RTI views are typically composed of about **50-100** photos, and the Zenodo record exposes a **1.5 GB** supplementary RTI zip. | RTI / PTM style reflectance documentation and discussion of photogrammetry; not a full independent 3D scan benchmark. | CC BY 4.0 on Zenodo. | Useful for lighting and surface-detail representation, but RTI is closer to 2.5D reflectance documentation than COLMAP-style 3D reconstruction ground truth. |
| 4 | **Photogrammetry mesh-only proxy** | Magnani, Douglass, and Porter, 2016, "Three-Dimensional Models of Experimentally-Produced Lithic Artifacts Created using Expedient and Refined Photogrammetry Protocols," DOI **10.13020/D6T88N** ([ranked_dataset_table[8]] [28]) | Data Repository for the University of Minnesota | Count not explicitly stated in extracted record; models are distributed in four zip files | Source photos not included in the extracted record. Protocols include expedient photogrammetry with wire stand and auto mode, and refined photogrammetry with turntable, controlled lighting, and manual settings. | Photogrammetry models processed with Agisoft PhotoScan and Geomagic Design X; OBJ, MTL, and JPG texture files. | Subject to University Digital Conservancy terms of use. | Good for comparing mesh-analysis outputs across model-quality regimes, but not usable as COLMAP input because image sets are absent. |
| 5 | **Best scan-only ground-truth pool** | Falcucci, Moroni, Negrino, Peresani, and Riel-Salvatore, 2025, "The Open Aurignacian Project: 3D scanning and the digital preservation of the Italian Paleolithic record" ([ranked_dataset_table[9]] [89]); related research compendium on Zenodo ([ranked_dataset_table[10]] [23]) | Nature Scientific Data; four Zenodo repositories | **2,016** stone tools from Grotta di Fumane, Grotta di Castelcivita, Grotta della Cala, and Riparo Bombrini | No raw multi-angle photo sets reported. | Artec Space Spider for **1,250** models, micro-CT for **571**, Artec Micro for **195**. PLY files, with selected earlier Grotta di Fumane models in WRL. | CC BY 4.0. | Excellent geometric reference pool, but not a photogrammetry benchmark without matching photographs. Micro-CT scans for very fine details are reported as slightly lower resolution than Artec Micro. |
| 6 | **Scan-only experimental flake-scar mesh set** | Liu, Valletta, Baena Preysler, and Falcucci, 2026, "3D Lithic Meshes and Metadata for 'Reconstructing the sequentiality of adjacent flake removal scars on lithic 3D models: A curvature-based computational approach'," DOI **10.5281/zenodo.18261895** ([ranked_dataset_table[11]] [51]) | Zenodo | **137** experimentally produced stone-tool meshes | Photos not available in the record. | Artec Micro I and Artec Space Spider I; PLY meshes. | CC BY 4.0. | Good for downstream curvature/scar algorithms and as possible ground truth if new photos can be acquired; not usable as image-input benchmark now. |
| 7 | **Scan/protocol background, mostly absorbed by Open Aurignacian** | Goldner, Karakostis, and Falcucci, 2022, "Practical and technical aspects for the 3D scanning of lithic artefacts using micro-computed tomography techniques and laser light scanners... Introducing the StyroStone protocol," DOI **10.1371/journal.pone.0267163** ([ranked_dataset_table[12]] [16]) | PLOS ONE; associated Zenodo context | Several hundred bladelets and blades from Fumane Cave are discussed in extracted analysis | No source photo sets reported. | Micro-CT and Artec structured-light scanning; extracted analysis reported PLY and WRL formats in associated data context. | Open-access article; associated model access should be verified record by record. | Important method paper for small lithics, but it is not a paired photo benchmark. |
| 8 | **Nubian Levallois OSF: not currently usable from public osfstorage roots** | "Quantifying Levallois: a 3D geometric morphometric approach to Nubian technology," DOI **10.17605/OSF.IO/SJ8ZV** ([ranked_dataset_table[13]] [4]); parent NUBIAN project ([ranked_dataset_table[14]] [3]) | OSF; linked GitHub mentioned in OSF metadata | Unknown from public file roots | No photos visible in the inspected osfstorage roots. | No meshes visible in the inspected osfstorage roots. API file lists for `sj8zv` and `xz7cb` returned `data: []` and `total: 0` ([ranked_dataset_table[15]] [40], [ranked_dataset_table[16]] [41]). | Public OSF project page; file access may depend on linked storage/components not visible in root osfstorage. | Do not treat it as a bulk-download photo/mesh dataset until component/provider traversal confirms files. |
| 9 | **RTI-only educational data** | "RTI Files," Reflectance Transformation Imaging For Lithics ([ranked_dataset_table[17]] [95]) | Standalone website | 3 lithic points: modern obsidian, chalcedony, and Molina Spring Clovis | Underlying source-photo counts not provided on extracted page. | RTI/PTM files up to 70 MB; requires RTIViewer. | Freely accessible; explicit reuse license not extracted. | Useful for reflectance and lighting experiments, but not a COLMAP photogrammetry dataset and not a 3D mesh ground truth set. |
| 10 | **Synthetic / ML adjacent, not a rendered photogrammetry benchmark** | "A proof of concept for machine learning-based virtual knapping" ([ranked_dataset_table[18]] [50]) | Scientific Reports / PMC; code/data linked from article | Programmatically generated virtual lithic data; extracted analysis noted 1,000 synthetic flakes and 1,000 non-flakes | No public multi-view rendered photo set found. | Synthetic 3D / ML virtual knapping workflow, not paired rendered image sequences for COLMAP validation. | Article is open; project data/code are described as available through OSF in the extracted analysis. | Useful conceptual precedent for synthetic Dibble data, but not directly usable as a photogrammetry image benchmark. |

The table shows a clear split: the best image datasets lack independent ground truth, while the best ground-truth meshes lack source photographs. For Dibble, the fastest practical route is to combine the Giant Handaxe and ground-stone photo sets for pipeline smoke tests, while using Open Aurignacian and other scan-only mesh sets to design a new controlled photo-capture campaign.

## Case Studies and Practical Implications

### Giant Handaxe: strong COLMAP input, weak validation target

The Maritime Academy Giant Handaxe is the closest match to Dibble's immediate pipeline need because it supplies **662 RAW DNG images** of one lithic artifact across four records ([case_studies_and_practical_implications[0]] [45], [case_studies_and_practical_implications[1]] [47], [case_studies_and_practical_implications[2]] [69], [case_studies_and_practical_implications[3]] [71]). Its associated Sketchfab model gives you a target mesh to compare against, but the public evidence points to a photogrammetric product rather than a CT or structured-light scan ([case_studies_and_practical_implications[4]] [48]).

For Dibble, that means the Giant Handaxe is best used to test ingestion, masking, image ordering, COLMAP parameterization, dense reconstruction, and mesh cleaning. It should not be used to report absolute geometric accuracy unless you can obtain an independent scan from the curators or a metrically validated reference model.

### Open Aurignacian: excellent ground truth trapped without photos

The Open Aurignacian Project is the opposite case. It has scale, curation, and high-quality geometry: **2,016** stone tools, with explicit scanner breakdowns of **1,250** Artec Space Spider, **571** micro-CT, and **195** Artec Micro models ([case_studies_and_practical_implications[5]] [89]). The models are open, documented, and distributed through Zenodo, with PLY as the main format and CC BY 4.0 licensing.

The problem is that the source photos are absent because the capture method was scanning, not photogrammetry. The most valuable Dibble partnership would be to photograph a stratified subset of these exact artifacts under a reproducible turntable protocol, then use the existing PLY scans as ground truth. That would produce the benchmark the public repositories currently lack.

### Ground-stone photogrammetry: protocol benchmark rather than lithic ground truth

Sorrentino et al. provide unusually detailed capture protocols: Nikon D750 RAW images, controlled object rotation, multiple camera heights, and two strategies that yield **100-120** or **150-190** images per stage ([case_studies_and_practical_implications[6]] [83]). Their dataset is valuable because it documents repeat acquisition and morphometric change, not just a finished mesh.

The limitation is scope and independence. The objects are ground-stone replicas, not knapped flint flakes or cores, and the models are photogrammetric outputs. Use this dataset to tune COLMAP robustness across lighting, orientation, and repeated states, but keep it separate from a strict CT/structured-light accuracy benchmark.

## Repository Audit

| Repository | Search focus | Relevant hits found | Outcome for Dibble |
|---|---|---|---|
| OSF | "lithic photogrammetry", "stone tool 3D", "Levallois", Nubian Levallois GUIDs | Nubian parent `xz7cb` and component `sj8zv`; osfstorage roots returned zero files ([repository_audit[0]] [40], [repository_audit[1]] [41]) | No verified public photo+mesh payload from OSF roots. Traverse children/providers before concluding a project is empty. |
| Zenodo | "lithic", "photogrammetry", "stone tool", "handaxe", "PLY", "OBJ" | Open Aurignacian, Sorrentino ground stones, Looten et al. RTI/representation data, Liu et al. 3D Lithic Meshes ([repository_audit[2]] [89], [repository_audit[3]] [83], [repository_audit[4]] [51]) | Best overall repository for lithic 3D assets, but pairings are missing: either images or independent scans are absent. |
| UCL RDR / Figshare | Giant Handaxe RAW image records | Four RAW-image zip records totaling 662 DNG images ([repository_audit[5]] [45]) | Best immediate COLMAP input set; one artifact only and no independent scan. |
| Sketchfab | "handaxe photogrammetry", "lithic photogrammetry" | Maritime Academy Giant Handaxe model and other isolated lithic models ([repository_audit[6]] [48]) | Useful for visual reference and occasional downloadable meshes, but source photos and metrology are usually absent. |
| MorphoSource | "lithic", "flint", "stone tool", "handaxe", "flake" | Search surfaced only general MorphoSource pages; no specific paired lithic photo+scan record was found. MorphoSource generally hosts high-resolution 3D data including raw microCT and surface meshes ([repository_audit[7]] [58]) | Possible hidden opportunity, but no public paired lithic benchmark found through indexed search. Manual MorphoSource search or API access may be needed. |
| tDAR | "lithic 3D" | Only repository landing pages surfaced in search results ([repository_audit[8]] [66]) | No relevant paired dataset found. |
| ADS | "3D lithic", "photogrammetry lithic" | Only general ADS pages surfaced in search results ([repository_audit[9]] [11]) | No relevant paired dataset found. |
| Open Context | "lithic 3D", "handaxe 3D model" | Only general Open Context pages surfaced; Open Context states it publishes archaeological data, images, maps, field notes, and 3D models ([repository_audit[10]] [65]) | No specific paired lithic benchmark found. |
| GitHub | "lithic photogrammetry photos mesh", `jmcascalheira/nubian3D` | No repository pairing source photos with lithic meshes found; osfclient is relevant for downloading OSF projects ([repository_audit[11]] [8]) | Useful tooling source, not a dataset source in this search. |

## Critical Gaps

The public data gap is structural. Archaeology repositories increasingly preserve **3D meshes** because they are compact, reusable, and citation-friendly, while source photographs are large and often treated as intermediate processing files. Conversely, the few public lithic photo sets are usually photogrammetry teaching or outreach datasets, so their 3D models are derived from the same images and cannot serve as independent ground truth. For Dibble, the highest-value next step is not more scraping; it is a targeted partnership: photograph a subset of Open Aurignacian or similar scan-only artifacts under a documented COLMAP-friendly protocol, then release the image sets plus existing PLY scans as a benchmark.

## OSF Bulk Downloading Procedure

### osfclient command syntax

The official `osfclient` user guide documents these core commands ([osf_bulk_downloading_procedure[0]] [73]):

```bash
pip install osfclient

# List files in a public OSF project or component by GUID
osf -p sj8zv list
# Python 3 installations may also support:
osf -p sj8zv ls

# Clone all files from a project/component into a local directory
osf -p sj8zv clone ./sj8zv_download

# Fetch a single file by remote OSF path
osf -p sj8zv fetch remote/path/file.ext ./local/file.ext
```

For the Nubian Levallois GUIDs inspected here, `sj8zv` and `xz7cb` public osfstorage roots returned no files through the API, so `clone` may legitimately produce an empty download unless files are in child components or linked providers ([osf_bulk_downloading_procedure[1]] [40], [osf_bulk_downloading_procedure[2]] [41]).

### Direct API and files.osf.io approach

The robust approach is:

1. Get node metadata: `https://api.osf.io/v2/nodes/{GUID}/`.
2. List child components: `https://api.osf.io/v2/nodes/{GUID}/children/`.
3. List storage providers: `https://api.osf.io/v2/nodes/{GUID}/files/`.
4. List provider contents, commonly: `https://api.osf.io/v2/nodes/{GUID}/files/osfstorage/`.
5. Follow `links.next` for pagination.
6. For folders, follow `relationships.files.links.related.href`.
7. For files, download `links.download`. Those URLs are the direct file-download route and usually resolve through `files.osf.io/v1/`.

A top-level-only shell example is:

```bash
GUID=sj8zv
curl -s "https://api.osf.io/v2/nodes/${GUID}/files/osfstorage/" > osf_root.json
jq -r '.data[] | select(.attributes.kind == "file") | .links.download' osf_root.json |
while read -r url; do
  curl -L -OJ "$url"
done
```

That shell pattern does **not** handle nested folders or pagination by itself. Use the Python script below for real repositories.

### Large OSF repository caveats

The OSF API documentation states that API requests are rate limited and that result sets are paginated ([osf_bulk_downloading_procedure[3]] [72]). I did not find an official public rule saying ">10 GB" or ">1,000 files" is a hard failure threshold. Practical workarounds are:

- Prefer API traversal over the web UI for large repositories.
- Follow `links.next` until it is null.
- Retry `429`, `500`, `502`, `503`, and `504` with exponential backoff.
- Respect `Retry-After` when OSF sends it.
- Stream downloads to disk instead of loading files into memory.
- Skip files already downloaded with the expected size.
- Split very large projects by OSF component or provider.
- Use authenticated requests with `OSF_TOKEN` when files are private or when you need more stable access.
- Be cautious with third-party whole-project ZIP tools. The PyPI `osf-downloader` package supports `osf-download download <OSF_ID> ./data`, but its own page notes no OSF authentication support, `osfstorage`-only support, and possible time/memory issues for very large projects ([osf_bulk_downloading_procedure[4]] [99]).

### Ready-to-run Python downloader

Save as `download_osf_project.py`, then run:

```bash
pip install requests
python download_osf_project.py sj8zv ./osf_downloads
# Optional for private or restricted projects:
OSF_TOKEN=your_token python download_osf_project.py sj8zv ./osf_downloads
```

```python
#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests

API = "https://api.osf.io/v2"
RETRY_STATUS = {429, 500, 502, 503, 504}


def clean_name(value):
    value = value or "untitled"
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip()
    return value[:150] or "untitled"


def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "dibble-osf-bulk-downloader/1.0"})
    token = os.environ.get("OSF_TOKEN")
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def request_with_retries(session, method, url, **kwargs):
    delay = 2.0
    for attempt in range(8):
        response = session.request(method, url, timeout=120, **kwargs)
        if response.status_code not in RETRY_STATUS:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            sleep_for = float(retry_after)
        else:
            sleep_for = delay
            delay = min(delay * 2.0, 120.0)
        print(f"Retryable HTTP {response.status_code} for {url}; sleeping {sleep_for:.1f}s", file=sys.stderr)
        time.sleep(sleep_for)
    response.raise_for_status()
    return response


def get_json(session, url):
    return request_with_retries(session, "GET", url).json()


def paginated_items(session, url):
    while url:
        payload = get_json(session, url)
        for item in payload.get("data", []):
            yield item
        url = payload.get("links", {}).get("next")


def download_file(session, download_url, destination, expected_size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        print(f"SKIP {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"GET  {destination}")
    response = request_with_retries(session, "GET", download_url, stream=True, allow_redirects=True)
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp.replace(destination)


def walk_folder(session, folder_api_url, local_dir):
    for item in paginated_items(session, folder_api_url):
        attrs = item.get("attributes", {})
        links = item.get("links", {})
        name = clean_name(attrs.get("name") or item.get("id"))
        kind = attrs.get("kind")
        size = attrs.get("size")

        if kind == "folder":
            related = (
                item.get("relationships", {})
                .get("files", {})
                .get("links", {})
                .get("related", {})
                .get("href")
            )
            if related:
                walk_folder(session, related, local_dir / name)
            continue

        download_url = links.get("download")
        if download_url:
            download_file(session, download_url, local_dir / name, expected_size=size)


def provider_roots(session, guid):
    providers_url = f"{API}/nodes/{guid}/files/"
    roots = []
    try:
        providers = list(paginated_items(session, providers_url))
    except requests.HTTPError:
        providers = []

    for provider in providers:
        attrs = provider.get("attributes", {})
        provider_name = clean_name(attrs.get("provider") or attrs.get("name") or provider.get("id"))
        related = (
            provider.get("relationships", {})
            .get("files", {})
            .get("links", {})
            .get("related", {})
            .get("href")
        )
        if related:
            roots.append((provider_name, related))

    if not roots:
        roots.append(("osfstorage", f"{API}/nodes/{guid}/files/osfstorage/"))
    return roots


def download_node(session, guid, output_root, include_children=True):
    node = get_json(session, f"{API}/nodes/{guid}/")
    attrs = node.get("data", {}).get("attributes", {})
    title = clean_name(attrs.get("title") or guid)
    node_dir = output_root / f"{title}_{guid}"
    node_dir.mkdir(parents=True, exist_ok=True)

    print(f"NODE {guid}: {title}")
    for provider_name, root_url in provider_roots(session, guid):
        print(f"PROVIDER {provider_name}: {root_url}")
        try:
            walk_folder(session, root_url, node_dir / provider_name)
        except requests.HTTPError as exc:
            print(f"WARN provider failed for {guid} {provider_name}: {exc}", file=sys.stderr)

    if include_children:
        children_url = f"{API}/nodes/{guid}/children/"
        for child in paginated_items(session, children_url):
            child_id = child.get("id")
            if child_id:
                download_node(session, child_id, node_dir / "components", include_children=True)


def main():
    parser = argparse.ArgumentParser(description="Recursively download a public OSF project/component by GUID.")
    parser.add_argument("guid", help="OSF node GUID, for example sj8zv")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--no-children", action="store_true", help="Do not traverse child components")
    args = parser.parse_args()

    session = make_session()
    download_node(session, args.guid, Path(args.output), include_children=not args.no_children)


if __name__ == "__main__":
    main()
```

## Synthesis

Dibble needs a benchmark with four properties at once: many images, the same physical artifact, independent high-quality geometry, and reusable licensing. The public record currently offers these properties in separate silos. Giant Handaxe and the ground-stone study have the images, but their 3D targets are photogrammetry outputs. Open Aurignacian and Liu et al. have high-quality scanner-derived meshes, but no photos. RTI datasets have rich surface lighting information, but not COLMAP-ready geometry.

The non-obvious implication is that the best benchmark strategy is hybrid. Use photo-rich proxies to harden the COLMAP pipeline, use scan-only repositories to define target artifact classes and ground-truth mesh standards, then create a small new paired capture campaign around artifacts that already have open scanner meshes. A 50-100 artifact paired subset from Open Aurignacian-style collections would be more valuable for Dibble validation than thousands of unpaired meshes.

## References

1. *Levallois-Perret (Town/city) • Mapy.com*. http://mapy.com/en?id=63080&source=osm
2. *Quantifying Levallois: a 3D geometric morphometric approach ...*. https://link.springer.com/article/10.1007/s12520-025-02199-2
3. *OSF | NUBIAN: Nubian Levallois technology, behavioural ...*. https://osf.io/xz7cb/wiki/home/
4. *OSF | Quantifying Levallois: a 3D geometric morphometric ...*. https://osf.io/sj8zv/
5. *(PDF) Quantifying Levallois: a 3D geometric morphometric ...*. https://www.researchgate.net/publication/390135025_Quantifying_Levallois_a_3D_geometric_morphometric_approach_to_Nubian_technology
6. *OSF API: The Complete Guide*. http://zuplo.com/learning-center/osf-api
7. *GitHub - mkchenxi/scraping_osf: This script is a batch OSF ...*. https://github.com/mkchenxi/scraping_osf
8. *GitHub - osfclient/osfclient: A python library and command ...*. https://github.com/osfclient/osfclient
9. *GitHub - aselimc/osf-downloader: Download files from osf.io*. https://github.com/aselimc/osf-downloader
10. *Welcome to OSF Projects*. https://help.osf.io/article/353-welcome-to-projects
11. *Archaeology Data Service*. https://archaeologydataservice.ac.uk/
12. *Zenodo*. https://zenodo.org/
13. *MorphoSource*. https://www.morphosource.org/browse/projects?locale=en
14. *ADS 3D Viewer: a 3D Real-Time System for the Management and ...*. https://cordis.europa.eu/project/id/625636/reporting
15. *Flint handaxe - Sizewell C, Suffolk - 3D model by Oxford ...*. https://sketchfab.com/3d-models/flint-handaxe-sizewell-c-suffolk-b3ffd4e89e374c7db86f7db09dbf7970
16. *Practical and technical aspects for the 3D scanning of lithic ...*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267163
17. *Artifact3-D: New software for accurate, objective and ... - PLOS*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0268401
18. *Three-Dimensional Modelling and Visualization of Stone ... - MDPI*. https://www.mdpi.com/2673-4591/27/1/35
19. *3D models of lithic artefacts: A test on their efficacy*. https://www.sciencedirect.com/science/article/pii/S2212054823000243
20. *fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA · Hugging Face*. https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
21. *Senior Software Engineer, Data Platform @ Lithic*. http://jobright.ai/jobs/info/69e746d77820c036924d56e0?visit=senior-software-engineer,-platforms-fde-jobs-in-united-states
22. *PROCESSING GUIDELINES FOR PHOTOGRAMMETRY - Zenodo*. https://zenodo.org/records/7940422/files/Processing_Version_3.0_240523.pdf?download=1
23. *Research Compendium for 'The Open Aurignacian Project: 3D ...*. https://zenodo.org/records/15397965
24. *Zenodo*. http://zenodo-rdm.web.cern.ch/
25. [GitHub - awsaf49/artifact: [ICIP 2023] ArtiFact: A Large ...](https://github.com/awsaf49/artifact)
26. *Dataset Search*. https://datasetsearch.research.google.com/
27. *Reflectance spectroscopy in combination with cluster analysis ...*. https://www.sciencedirect.com/science/article/pii/S2352409X21002534
28. *Three-Dimensional Models of Experimentally-Produced Lithic ...*. https://conservancy.umn.edu/items/42498dea-f904-43a5-8cdc-9aab61a82dd9
29. *Towards a more robust representation of lithic industries in ...*. https://peercommunityjournal.org/articles/10.24072/pcjournal.562/
30. *Item - Zip file containing RAW format image files used to ...*. https://rdr.ucl.ac.uk/articles/dataset/Zip_file_containing_RAW_format_image_files_used_to_capture_the_Giant_Handaxe_Files_552-662_4_4_/23280374
31. *photogrammetry_datasets • open-archaeo*. https://open-archaeo.info/post/photogrammetry-datasets/
32. *Photogrammetry datasets: 100% free to download*. https://www.gea.scanbim.ch/posts/dataset-scanbim-geomatik-vermessung-zurich-list-of-photogrammetry-datasets-downloadable/
33. *Transforming lithic representation: The potential of ...*. https://archaeo.peercommunityin.org/articles/rec?id=605
34. *Lithic - GitHub*. http://github.com/lithic-com
35. *Artifact3-D download | SourceForge.net*. https://sourceforge.net/projects/artifact3-d/
36. *OSF*. https://osf.io/
37. *Center for Open Science - Wikipedia*. http://en.wikipedia.org/wiki/Center_for_Open_Science
38. *How To Photograph Lithic Artifacts | Kait Photography*. https://kaitphotography.com.au/h-photography/how-to-photograph-lithic-artifacts.html
39. *OSF Digital*. http://osf.digital/insights/success-stories
40. *Node Files List – Django REST framework*. https://api.osf.io/v2/nodes/sj8zv/files/osfstorage/
41. *Node Files List – Django REST framework*. https://api.osf.io/v2/nodes/xz7cb/files/osfstorage/
42. *Node Detail – Django REST framework*. https://api.osf.io/v2/nodes/sj8zv/
43. *Node Detail – Django REST framework*. https://api.osf.io/v2/nodes/xz7cb/
44. *GitHub - jmcascalheira/nubian3D · GitHub*. https://github.com/jmcascalheira/nubian3D
45. *Item - Zip file containing RAW format image files used to ...*. https://rdr.ucl.ac.uk/articles/dataset/Zip_file_containing_RAW_format_image_files_used_to_capture_the_Giant_Handaxe_Files_001-0160_1_4_/23591925
46. *Put Your Generosity To Work – Community Foundation Santa Cruz County*. http://cfscc.org/finance-and-governance
47. *Zip file containing RAW format image files used to capture ...*. https://figshare.com/articles/dataset/Zip_file_containing_RAW_format_image_files_used_to_capture_the_Giant_Handaxe_Files_161-336_2_4_/22957319/1
48. *Maritime Academy Giant Handaxe - 3D model by sarahMduffy.uk*. https://sketchfab.com/3d-models/maritime-academy-giant-handaxe-242e16a1e43e4a16bc2bfcdfbe3cdc59
49. *A proof of concept for machine learning-based virtual ...*. https://www.nature.com/articles/s41598-021-98755-6
50. *A proof of concept for machine learning-based virtual ...*. https://pmc.ncbi.nlm.nih.gov/articles/PMC8497608/
51. *3D Lithic Meshes and Metadata for "Reconstructing the ...*. https://zenodo.org/records/18261895
52. *The Open Aurignacian Project: 3D scanning and the digital ...*. https://www.researchgate.net/publication/392838964_The_Open_Aurignacian_Project_3D_scanning_and_the_digital_preservation_of_the_Italian_Paleolithic_record
53. *The Open Aurignacian Project: 3D scanning and the digital ...*. https://www.nature.com/articles/s41597-025-05330-z.pdf
54. *The Open aurignacian Project: 3D scanning and the digital ...*. https://unige.iris.cineca.it/handle/11567/1254837
55. *The Open Aurignacian Project: 3D scanning and the digital ...*. https://ui.adsabs.harvard.edu/abs/2025NatSD..12.1037F/abstract
56. *MorphoSource · bio.tools*. https://bio.tools/morphosource
57. *How to Create and Edit Media - MorphoSource Documentation ...*. https://duke.atlassian.net/wiki/spaces/MD/pages/35422701/How+to+Create+and+Edit+Media
58. *MorphoSource.org | Functional and Evolutionary Morphology ...*. https://femr.la.psu.edu/research/morphosource-org/
59. *MorphoSource - GitHub*. https://github.com/MorphoSource
60. *photogrammetry · GitHub Topics · GitHub*. https://github.com/topics/photogrammetry
61. *Robin Gandhi - Lithic*. http://linkedin.com/in/robingandhi
62. *open-archaeo*. https://open-archaeo.info/
63. *Matrix3D: Large Photogrammetry Model All-in-One*. https://nju-3dv.github.io/projects/matrix3d/
64. *Howard Tyson - Head of Engineering at Lithic | LinkedIn*. http://linkedin.com/in/howardtyson
65. *Open Context: Publisher of Research Data*. https://opencontext.org/
66. *Welcome to the Digital Archaeological Record*. https://core.tdar.org/
67. *Privacy.com raises $43m and rebrands as Lithic*. http://fintechfutures.com/venture-capital-funding/privacy-com-raises-43m-and-rebrands-as-lithic
68. *Alix Odendhal - Product Lithic | Ex-Adyen, Ex-Marqeta*. http://linkedin.com/in/alixodendhal
69. *Zip file containing RAW format image files used to capture ...*. https://figshare.com/articles/dataset/Zip_file_containing_RAW_format_image_files_used_to_capture_the_Giant_Handaxe_Files_337-551_3_4_/22959137/1
70. *Zip file containing RAW format image files used to capture ...*. https://b2find.eudat.eu/dataset/37156e3f-b671-5fb6-aad6-9b10c1e591d3
71. *Zip file containing RAW format image files used to capture ...*. https://figshare.com/articles/dataset/Zip_file_containing_RAW_format_image_files_used_to_capture_the_Giant_Handaxe_Files_552-662_4_4_/23280374/1
72. *OSF APIv2 Documentation*. https://developer.osf.io/
73. *User’s Guide — osfclient 0.0.3 documentation*. http://osfclient.readthedocs.io/en/latest/cli-usage.html
74. *osf-api-v2-typescript/docs/plans/implemenetation-plan-issue ...*. https://github.com/hirakinii/osf-api-v2-typescript/blob/main/docs/plans/implemenetation-plan-issue-62.md
75. *osfclient · PyPI*. https://pypi.org/project/osfclient/
76. *GitHub - sulemvn/3D-CGAN: This project explores the ...*. https://github.com/sulemvn/3D-CGAN
77. *Tafi - LinkedIn*. http://linkedin.com/company/maketafi
78. *An integrated machine learning framework using borehole ...*. https://www.sciencedirect.com/science/article/pii/S0013795225001462
79. *MS-CGAN: Fusion of conditional generative adversarial ...*. https://www.sciencedirect.com/science/article/pii/S0926985124002477
80. *University Digital Conservancy :: Home*. https://conservancy.umn.edu/
81. *Three-Dimensional Models of Experimentally-Produced Lithic ...*. https://www.semanticscholar.org/paper/Three-Dimensional-Models-of-Experimentally-Produced-Magnani-Douglass/a0c762f83aa759387159a8f6d93e480ad7db1d1d
82. *api.datacite.org*. https://api.datacite.org/dois/application/vnd.datacite.datacite+json/10.13020/d6t88n
83. *Close-range photogrammetry reveals morphometric changes on ...*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289807
84. *GitHub - natowi/photogrammetry_datasets: Collection of 350 ...*. https://github.com/natowi/photogrammetry_datasets
85. *Photogrammetric Documentation of Stone Surface Topography ...*. https://www.mdpi.com/2075-5309/13/2/439
86. *How to Validate Photogrammetry and LiDAR Outputs*. https://anvil.so/post/how-to-validate-photogrammetry-and-lidar-outputs
87. *Example projects - real photogrammetry data - Pix4D*. https://support.pix4d.com/hc/en-us/articles/360000235126
88. *The Open Aurignacian Project. Volume 3: Grotta della Cala in ...*. https://zenodo.org/records/15383121
89. *The Open Aurignacian Project: 3D scanning and the digital ...*. https://www.nature.com/articles/s41597-025-05330-z
90. *The Open Aurignacian Project - Armando Falcucci*. https://www.armandofalcucci.com/project/open_aurignacian/
91. *The Open Aurignacian Project: 3D scanning and the digital ...*. https://www.armandofalcucci.com/publication/falcucci-et-al_2025_scidata/
92. *Downloading Data from OSF Projects Using R (OSF API)*. https://codewithsusan.com/notes/r-download-data-from-osf
93. *Archaeology Data Service - The Digital Classicist Wiki*. https://wiki.digitalclassicist.org/Archaeology_Data_Service
94. *Aerial Metrics Home*. http://aerial-metrics.com/
95. *RTI Files | Reflectance Transformation Imaging For Lithics*. https://rtimage.us/?page_id=18
96. *Close-range photogrammetry reveals morphometric changes on ...*. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10443871/
97. *OSF APIv2 Documentation*. https://developer.osf.io/?ref=public-apis
98. *Welcome to osfclient’s documentation! — osfclient 0.0.3 ...*. https://osfclient.readthedocs.io/
99. *osf-downloader · PyPI*. https://pypi.org/project/osf-downloader/
100. *Advancing Face-to-Face Emotion Communication: A Multimodal Dataset (AFFEC)*. http://arxiv.org/html/2504.18969v2
101. *Galaxy*. http://phage.usegalaxy.eu/datasets/26c75dcccb616ac85cdadcd6f78a1da5/details
102. *Sites for 3D Assets Every Designer Should Know. 2D is ... - Instagram*. http://instagram.com/p/DXCXzqWE-69
103. *Handaxe-photogrammetry 3D models - Sketchfab*. https://sketchfab.com/tags/handaxe-photogrammetry
104. *Lithic - Download Free 3D model by USC SCIAA ... - Sketchfab*. https://sketchfab.com/3d-models/lithic-507542341ab54f17bd47f98217d16fac
105. *Towards a more robust representation of lithic industries in archaeology: a critical review of traditional approaches and modern techniques*. https://doi.org/10.5281/zenodo.15411558
