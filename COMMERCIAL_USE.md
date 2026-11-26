# Commercial-safe setup (fork notes)

This fork removes the non-commercial dependencies from the image-to-GLB path so
the assets it produces can be used in a paid project. This is an engineering
note, not legal advice.

## What changed

| Area | Before | After |
|---|---|---|
| GLB texture baking (`o-voxel/o_voxel/postprocess.py`) | `nvdiffrast` UV rasterization (NVIDIA Source Code License, non-commercial) | `o_voxel/uv_raster.py`, pure PyTorch |
| Texturing pipeline (`trellis2/pipelines/trellis2_texturing.py`) | `nvdiffrast` | `o_voxel/uv_raster.py` |
| Background removal (`trellis2/pipelines/rembg/BiRefNet.py`) | Whatever the pipeline config names (may be `briaai/RMBG-2.0`, CC BY-NC 4.0) | Any `briaai/*` model is replaced with `ZhengPeng7/BiRefNet` (MIT) |

## How to install

Install as in the main README but leave out `--nvdiffrast` and `--nvdiffrec`:

```sh
. ./setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm
```

Then confirm nothing non-commercial is present:

```sh
python check_commercial_deps.py
```

## How to generate

```sh
python example_commercial.py path/to/image.png output.glb
```

Do **not** use `example.py`, `app.py`, `app_texturing.py` or
`trellis2.utils.render_utils` / `trellis2.renderers`: the preview renderers
still use nvdiffrast and nvdiffrec. Preview the GLB in Blender or Roblox Studio.

## Remaining licenses on the generation path

TRELLIS.2 code and weights (MIT), o-voxel (MIT), CuMesh and FlexGEMM (MIT),
utils3d (MIT), trimesh (MIT), BiRefNet (MIT), DINOv3 (Meta DINOv3 License,
permits commercial use with conditions; read it once).

## Tests

```sh
python -m pytest o-voxel/tests/test_uv_raster.py   # CPU only
```
