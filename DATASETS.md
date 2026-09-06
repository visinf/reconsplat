## Datasets

Our models are trained on the following datasets:
- **[RealEstate10K](https://google.github.io/realestate10k/)** — real-estate
  video tours, mainly indoor.
- **[DL3DV-10K](https://github.com/DL3DV-10K/Dataset)** — (a split with ~2K scenes) real-world scene captures, mainly outdoor. 

Using either one takes two steps: split the scenes into chunks, then process the scenes to obtain cameras and depth pseudo-labels.

### Getting the data

You may refer to [DepthSplat's `DATASETS.md`](https://github.com/cvg/depthsplat/blob/main/DATASETS.md) for how to obtain both datasets and a detailed pre-processing walkthrough of both. After pre-processing, point `dataset.roots` at the resulting folder (i.e., the parent folder containing the `train` and `test` splits).

> [!TIP]
> The defaults are `datasets/re10k` and `datasets/dl3dv`. If your data is somewhere else, link it there and you can leave the configs untouched:
> ```
> ln -s /path/to/re10k datasets/re10k
> ln -s /path/to/dl3dv datasets/dl3dv
> ```

### Processing the data

Depth supervision comes from pseudo-labels mined per scene. Run our script 
[`src/scripts/generate_pseudo_labels.py`](https://github.com/visinf/reconsplat/blob/main/src/scripts/generate_pseudo_labels.py)
(configured via [`config/labels.yaml`](https://github.com/visinf/reconsplat/blob/main/config/labels.yaml)) to obtain:

* **depth** — from [VGGT](https://github.com/facebookresearch/vggt) point maps, optionally refined
  with [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and aligned back
  to the VGGT scale by robust (Huber) regression on disparity, computing a single scale and offset per scene.
* **camera poses** — estimated by VGGT.

```bash
# Edit config/labels.yaml first: pipeline.load_dir (scene chunks), pipeline.out_dir (where labels go),
# pipeline.processor.dataset_name, and pipeline.video_depth.model_dir to use Video-Depth-Anything.
python -m src.scripts.generate_pseudo_labels

# VGGT only, if you want to skip the video-depth refinement stage (e.g., if you are only interested in cameras)
python -m src.scripts.generate_pseudo_labels_vggt_only
```

The pipeline writes cameras to `<out_dir>/<split>/cameras/` and depth to `<out_dir>/<split>/depths/`. After the processing, configure the loader to read it:

* `dataset.labels_root` and `dataset.cameras_root`: `<out_dir>`

Depth labels are only used for training and for depth metrics. Inference does not read them.

> [!TIP]
> Zarr is an alternative saving format that takes much less space. `src/scripts/quantize_depth.py`
> converts the depth `.npz` files into quantized Zarr directories. It reads `<root>/<split>/*.npz`.
> If you use it, set `dataset.labels_root` to its output and `dataset.load_labels_from_zarr_dirs` to
> `true`.

> [!IMPORTANT]
> The pipeline is resumable: it records progress under `pipeline.processor.progress_dir` and skips chunks listed in a cache file.

### Camera poses for evaluation

Evaluating a checkpoint requires **camera poses**, but not depth labels. Therefore, if you are evaluating a released checkpoint, you do **not** need to run the depth preprocessing described in [Processing the data](#processing-the-data). There are two ways to obtain camera poses without running VGGT yourself.

#### Option 1: VGGT cameras (recommended)

We release cached **VGGT-estimated camera poses** for the RE10K and DL3DV test splits. They are stored under `poses/re10k/test/` and `poses/dl3dv/test/` in the [ReconSplat Hugging Face repository](https://huggingface.co/gpstracquadanio/reconsplat).

Download them with:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p datasets   # same datasets/ directory as above; skip if you already symlinked it

# RE10K
hf download gpstracquadanio/reconsplat \
    --include "poses/re10k/*" \
    --local-dir datasets

# DL3DV
hf download gpstracquadanio/reconsplat \
    --include "poses/dl3dv/*" \
    --local-dir datasets
```

Then point the dataset loader to the downloaded poses and disable depth labels:

```bash
# RE10K
dataset.cameras_root=datasets/poses/re10k \
dataset.load_depth_labels=false
```

For DL3DV, replace `re10k` with `dl3dv`.

> [!NOTE]
> You can omit `dataset.load_depth_labels=false` if you have already generated the depth pseudo-labels described in [Processing the data](#processing-the-data). 

#### Option 2: COLMAP cameras

Both RE10K and DL3DV provide pre-computed COLMAP cameras, which can be used by setting:

```bash
dataset.load_other_cameras=false
```

Two additional adjustments are required:

- **Intrinsics.** The chunk files store normalized intrinsics, while the pipeline performs cropping in pixel space. The dataset loaders handle the required conversion automatically.
- **Scale.** COLMAP and VGGT camera poses generally use different scene scales, while **ReconSplat was trained with VGGT poses**. We therefore align the COLMAP camera centers to the VGGT ones with a per-scene transform.

The scale factors can be computed with:

```bash
python -m src.scripts.compute_colmap_pose_scales \
    --split test \
    --out assets/re10k_colmap_pose_scales_test.json
```

and passed to the loader through `dataset.scales_path`:

```bash
python -m src.main +experiment=re10k_diffusion_release mode=test \
    dataset.load_other_cameras=false \
    dataset.load_depth_labels=false \
    dataset.scales_path=assets/re10k_colmap_pose_scales_test.json \
    ...
```

`compute_colmap_pose_scales.py` uses the cached VGGT cameras from Option 1 as the scale reference.

### ScanNet++ (depth synthesis evaluation only)

We only use ScanNet++ for the depth synthesis evaluation, so no training data is needed. Download the iPhone NVS test split from [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/), then point `roots` in `config/dataset/scannet.yaml` at the folder that holds it.

The loader expects `nvs_test_iphone.txt` at that root, listing the test scenes, and reads each scene from `<scene>/iphone/` (`transforms.json`, `rgb/`, `depth/`).