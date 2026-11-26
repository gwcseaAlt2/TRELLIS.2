"""
Fails (exit code 1) if any non-commercial package used by TRELLIS / TRELLIS.2
previews is installed in the current Python environment.
Run this on the machine you generate game assets with.
"""
import importlib.util
import sys

NON_COMMERCIAL = {
    "nvdiffrast": "NVIDIA Source Code License (non-commercial)",
    "nvdiffrec_render": "NVIDIA Source Code License (non-commercial)",
    "diff_gaussian_rasterization": "Inria Gaussian-Splatting License (research only)",
    "diffoctreerast": "Inria-derived license (research only)",
}

found = [(m, lic) for m, lic in NON_COMMERCIAL.items() if importlib.util.find_spec(m) is not None]
if found:
    print("Non-commercial packages are installed:")
    for m, lic in found:
        print(f"  - {m}: {lic}")
    print("Uninstall them (pip uninstall ...) before generating assets for a paid project.")
    sys.exit(1)
print("OK: no known non-commercial rendering packages are installed.")
