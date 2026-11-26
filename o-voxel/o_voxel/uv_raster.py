"""
Pure-PyTorch UV-space rasterizer.

Replaces the nvdiffrast calls (``dr.rasterize`` + ``dr.interpolate``) that were
used to find, for every texel of the output texture, which triangle covers it
and the corresponding 3D surface position. nvdiffrast is licensed for
non-commercial use only; this module depends on PyTorch alone.

Conventions match the nvdiffrast call it replaces
(``uv * 2 - 1`` as clip-space x/y, OpenGL row order):
texel (row j, col i) has its center at uv = ((i + 0.5) / W, (j + 0.5) / H).
"""
from typing import Tuple
import torch


@torch.no_grad()
def rasterize_uv(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    max_candidates: int = 1 << 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize triangles given in UV space.

    Args:
        uvs: (V, 2) float UV coordinates in [0, 1].
        faces: (F, 3) int vertex indices into ``uvs``.
        resolution: texture width and height in texels.
        max_candidates: upper bound on candidate texels tested per chunk
            (controls peak memory).

    Returns:
        face_id: (H, W) int64, index of the covering face, -1 where empty.
        bary: (H, W, 3) float32 barycentric weights of the covering face.
    """
    device = uvs.device
    H = W = int(resolution)
    faces = faces.long()
    uvs = uvs.float()

    face_id = torch.full((H * W,), -1, dtype=torch.long, device=device)
    bary = torch.zeros((H * W, 3), dtype=torch.float32, device=device)
    if faces.shape[0] == 0:
        return face_id.view(H, W), bary.view(H, W, 3)

    # Triangle corners in pixel space (pixel centers at integer + 0.5).
    tri = uvs[faces] * torch.tensor([W, H], dtype=torch.float32, device=device)  # (F, 3, 2)
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])

    # Candidate texel range per face: centers c with lo <= c + 0.5 <= hi.
    lo = tri.min(dim=1).values
    hi = tri.max(dim=1).values
    x0 = torch.ceil(lo[:, 0] - 0.5).clamp(0, W - 1).long()
    y0 = torch.ceil(lo[:, 1] - 0.5).clamp(0, H - 1).long()
    x1 = torch.floor(hi[:, 0] - 0.5).clamp(-1, W - 1).long()
    y1 = torch.floor(hi[:, 1] - 0.5).clamp(-1, H - 1).long()
    bw = (x1 - x0 + 1).clamp(min=0)
    bh = (y1 - y0 + 1).clamp(min=0)
    valid = (area.abs() > 1e-12) & (bw > 0) & (bh > 0)
    counts = torch.where(valid, bw * bh, torch.zeros_like(bw))

    # Process faces in chunks so the expanded candidate list stays bounded.
    csum = torch.cumsum(counts, 0)
    total = int(csum[-1].item()) if csum.numel() else 0
    start_face = 0
    F = faces.shape[0]
    while start_face < F and total > 0:
        base = int(csum[start_face - 1].item()) if start_face > 0 else 0
        end_face = int(torch.searchsorted(csum, torch.tensor(base + max_candidates, device=device), right=True).item())
        end_face = max(end_face, start_face + 1)
        end_face = min(end_face, F)
        sl = slice(start_face, end_face)
        start_face = end_face

        c = counts[sl]
        n = int(c.sum().item())
        if n == 0:
            continue
        fid = torch.repeat_interleave(torch.arange(sl.start, sl.stop, device=device), c)
        offsets = torch.cumsum(c, 0) - c
        local = torch.arange(n, device=device) - torch.repeat_interleave(offsets, c)
        w = bw[fid]
        px = x0[fid] + local % w
        py = y0[fid] + local // w
        cx = px.float() + 0.5
        cy = py.float() + 0.5

        a, b, d = p0[fid], p1[fid], p2[fid]
        ar = area[fid]
        # Edge functions -> barycentrics (sign-normalized so either winding works).
        w0 = ((b[:, 0] - cx) * (d[:, 1] - cy) - (b[:, 1] - cy) * (d[:, 0] - cx)) / ar
        w1 = ((d[:, 0] - cx) * (a[:, 1] - cy) - (d[:, 1] - cy) * (a[:, 0] - cx)) / ar
        w2 = 1.0 - w0 - w1
        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        pix = (py * W + px)[inside]
        fid_in = fid[inside]
        bary_in = torch.stack([w0, w1, w2], dim=-1)[inside]
        # Where several faces cover one texel (shared edges, overlaps), pick a
        # single winner (highest face index) so face_id and bary stay consistent.
        win = torch.full((H * W,), -1, dtype=torch.long, device=device)
        win.scatter_reduce_(0, pix, fid_in, reduce="amax")
        keep = win[pix] == fid_in
        face_id[pix[keep]] = fid_in[keep]
        bary[pix[keep]] = bary_in[keep]

    return face_id.view(H, W), bary.view(H, W, 3)


def interpolate(
    attrs: torch.Tensor,
    faces: torch.Tensor,
    face_id: torch.Tensor,
    bary: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate per-vertex ``attrs`` (V, C) at every texel. Empty texels get 0.
    Returns (H, W, C).
    """
    mask = face_id >= 0
    out = torch.zeros((*face_id.shape, attrs.shape[-1]), dtype=attrs.dtype, device=attrs.device)
    f = faces.long()[face_id[mask]]                      # (N, 3)
    out[mask] = (attrs[f] * bary[mask].unsqueeze(-1).to(attrs.dtype)).sum(dim=1)
    return out
