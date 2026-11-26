"""
CPU tests for o_voxel.uv_raster (no GPU or compiled extensions needed).
Run: python -m pytest o-voxel/tests/test_uv_raster.py
"""
import importlib.util
import os
import torch

_spec = importlib.util.spec_from_file_location(
    "uv_raster", os.path.join(os.path.dirname(__file__), "..", "o_voxel", "uv_raster.py"))
uv_raster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uv_raster)


def _texel_centers(res):
    c = (torch.arange(res, dtype=torch.float32) + 0.5) / res
    v, u = torch.meshgrid(c, c, indexing="ij")  # row j -> v, col i -> u
    return u, v


def test_full_square_interpolates_uv_exactly():
    res = 64
    uvs = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    fid, bary = uv_raster.rasterize_uv(uvs, faces, res)
    assert (fid >= 0).all()
    pos = uv_raster.interpolate(uvs, faces, fid, bary)
    u, v = _texel_centers(res)
    assert torch.allclose(pos[..., 0], u, atol=1e-5)
    assert torch.allclose(pos[..., 1], v, atol=1e-5)


def test_row_order_matches_opengl_convention():
    # Triangle in the bottom-left half (v small) must land in low row indices.
    res = 32
    uvs = torch.tensor([[0., 0.], [1., 0.], [0., 0.25]])
    faces = torch.tensor([[0, 1, 2]])
    fid, _ = uv_raster.rasterize_uv(uvs, faces, res)
    rows = torch.nonzero(fid >= 0)[:, 0]
    assert rows.max() < res * 0.25


def _brute_force(uvs, faces, res):
    u, v = _texel_centers(res)
    p = torch.stack([u, v], -1).view(-1, 2)
    covered = torch.zeros(p.shape[0], dtype=torch.bool)
    for f in faces:
        a, b, c = uvs[f]
        def e(p0, p1):
            return (p1[0] - p0[0]) * (p[:, 1] - p0[1]) - (p1[1] - p0[1]) * (p[:, 0] - p0[0])
        e0, e1, e2 = e(b, c), e(c, a), e(a, b)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        covered |= inside
    return covered.view(res, res)


def test_random_triangles_match_brute_force_and_chunking():
    torch.manual_seed(0)
    res = 48
    n = 60
    uvs = torch.rand(n * 3, 2)
    faces = torch.arange(n * 3).view(n, 3)
    fid, bary = uv_raster.rasterize_uv(uvs, faces, res)
    ref = _brute_force(uvs, faces, res)
    # Allow a handful of disagreements exactly on edges (epsilon handling).
    assert ((fid >= 0) != ref).sum() <= 3
    # Barycentrics are valid and reproduce the texel center.
    m = fid >= 0
    assert torch.allclose(bary[m].sum(-1), torch.ones(int(m.sum())), atol=1e-5)
    pos = uv_raster.interpolate(uvs, faces, fid, bary)
    u, v = _texel_centers(res)
    assert torch.allclose(pos[m][:, 0], u[m], atol=1e-4)
    assert torch.allclose(pos[m][:, 1], v[m], atol=1e-4)
    # Tiny chunks give the same coverage.
    fid2, _ = uv_raster.rasterize_uv(uvs, faces, res, max_candidates=37)
    assert torch.equal(fid >= 0, fid2 >= 0)


def test_interpolates_3d_positions_and_both_windings():
    res = 16
    uvs = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    xyz = torch.tensor([[1., 2., 3.], [5., 2., 3.], [1., 6., 7.]])
    for faces in (torch.tensor([[0, 1, 2]]), torch.tensor([[0, 2, 1]])):
        fid, bary = uv_raster.rasterize_uv(uvs, faces, res)
        pos = uv_raster.interpolate(xyz, faces, fid, bary)
        u, v = _texel_centers(res)
        m = fid >= 0
        expect = torch.stack([1 + 4 * u, 2 + 4 * v, 3 + 4 * v], -1)
        assert torch.allclose(pos[m], expect[m], atol=1e-4)


def test_degenerate_and_empty():
    uvs = torch.tensor([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]])
    fid, _ = uv_raster.rasterize_uv(uvs, torch.tensor([[0, 1, 2]]), 8)
    assert (fid < 0).all()
    fid, _ = uv_raster.rasterize_uv(uvs, torch.zeros((0, 3), dtype=torch.long), 8)
    assert (fid < 0).all()
