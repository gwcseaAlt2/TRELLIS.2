"""
Image -> GLB without any non-commercial dependencies.

Unlike example.py, this script does not render a preview video (the preview
renderers use nvdiffrast / nvdiffrec, which are licensed for non-commercial
use only). Preview the exported GLB in Blender or Roblox Studio instead.

Usage:
    python example_commercial.py path/to/image.png output.glb
"""
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

image_path = sys.argv[1] if len(sys.argv) > 1 else "assets/example_image/T.png"
out_path = sys.argv[2] if len(sys.argv) > 2 else "sample.glb"

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open(image_path)
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)

glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export(out_path, extension_webp=True)
print(f"Saved {out_path}")
