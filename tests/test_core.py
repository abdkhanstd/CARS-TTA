"""Unit and integration tests. Run: pytest -q tests (needs data/cache/acdc for the data tests)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cars_tta import shifts  # noqa: E402
from cars_tta.common import make_source_split, patient_input  # noqa: E402
from cars_tta.data import crop_or_pad, load_patient, to_original, uncrop  # noqa: E402
from cars_tta.features import trajectory_features  # noqa: E402
from cars_tta.metrics import dice, patient_metrics  # noqa: E402
from cars_tta.tta import Episode, assert_restored  # noqa: E402
from cars_tta.unet import UNet2D  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "acdc")
has_data = os.path.exists(os.path.join(CACHE, "index.csv"))


def test_crop_pad_inverse():
    for h, w in [(300, 200), (180, 256), (256, 400)]:
        a = np.random.rand(3, h, w)
        c, rec = crop_or_pad(a)
        assert c.shape == (3, 256, 256)
        u = uncrop(c, rec)
        assert u.shape == a.shape
        hs, ws = max(0, (h - 256) // 2), max(0, (w - 256) // 2)
        np.testing.assert_allclose(u[:, hs:hs + min(h, 256), ws:ws + min(w, 256)], a[:, hs:hs + min(h, 256), ws:ws + min(w, 256)])


def test_split_exclusive():
    pids = [f"p{i}" for i in range(100)]
    s = make_source_split(pids, seed=1)
    assert set(s) == set(pids)
    assert {v for v in s.values()} == {"seg_train", "seg_val", "router_train", "router_val"}


@pytest.mark.parametrize("name", shifts.ALL)
def test_shift_preserves_shape_and_is_finite(name):
    rng = np.random.default_rng(0)
    v = np.random.rand(4, 64, 64).astype(np.float32)
    for sev in (0.2, 1.0, 1.5):
        o = shifts.apply_shift(v, [name], rng, sev)
        assert o.shape == v.shape and np.isfinite(o).all() and (o >= 0).all()
        assert o.std() > 0


def test_dice_edge_cases():
    a = np.zeros((2, 8, 8), np.uint8)
    assert dice(a, a, 1) == 1.0
    b = a.copy(); b[0, 0, 0] = 1
    assert dice(b, a, 1) == 0.0


def test_episode_restores_source_and_candidate0_is_source():
    torch.manual_seed(0)
    m = UNet2D(feats=(8, 16, 32))
    src = {k: v.clone() for k, v in m.state_dict().items()}
    ep = Episode(m, lr=1e-2, bs=4)
    ep.eata_margin = ep.sar_margin = 10.0  # untrained net: every slice is above the default reliability margin
    x = torch.randn(6, 1, 32, 32)
    with torch.no_grad():
        p0 = m(x).softmax(1).half().numpy()
    for method in ["tent", "eata", "sar"]:
        ep.method = method
        c, _, st = ep.run(x, K=3)
        assert len(c) == 4
        np.testing.assert_array_equal(c[0], p0)
        assert assert_restored(m, src)
        assert not np.array_equal(c[0], c[3])
    # a second patient sees exactly the same source model
    c2, _, _ = ep.run(x, K=1)
    np.testing.assert_array_equal(c2[0], p0)


def test_features_finite_with_empty_predictions():
    C = np.zeros((3, 4, 32, 32), np.float32); C[:, 0] = 1  # everything background
    f = trajectory_features([C, C, C, C], K=3)
    assert all(np.isfinite(v) for v in f.values())


@pytest.mark.skipif(not has_data, reason="ACDC cache not built")
def test_acdc_labels_and_geometry_roundtrip():
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    pat = load_patient(CACHE, idx.pid.iloc[0])
    for ph in pat.phases:
        assert set(np.unique(pat.lab[ph])) <= {0, 1, 2, 3}
        assert set(np.unique(pat.orig[ph])) == {0, 1, 2, 3}
        oh = np.eye(4, dtype=np.float32)[pat.lab[ph]].transpose(0, 3, 1, 2)
        back = to_original(oh, pat.geom[ph])
        assert back.shape == pat.orig[ph].shape
        m = patient_metrics({ph: back}, {ph: pat.orig[ph]}, pat.geom[ph]["spacing"])
        assert m["dice"] > 0.9, m  # resampling there and back keeps the label maps
    # LV (1) must be enclosed by myocardium (2): LV cavity centroid close to MYO centroid
    lab = pat.lab["ED"]
    assert np.linalg.norm(np.argwhere(lab == 1).mean(0) - np.argwhere(lab == 2).mean(0)) < 10


@pytest.mark.skipif(not has_data, reason="ACDC cache not built")
def test_patient_input_shapes():
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    pat = load_patient(CACHE, idx.pid.iloc[0])
    x, sl = patient_input(pat)
    assert x.shape[1:] == (1, 256, 256)
    assert sum(b - a for a, b in sl.values()) == x.shape[0]


def test_hist_match_maps_to_template():
    from cars_tta.candidates import QS, hist_match, source_template
    rng = np.random.default_rng(0)
    src = [rng.beta(2, 5, (4, 32, 32)).astype(np.float32) for _ in range(3)]
    tpl = source_template(src)
    tgt = rng.beta(5, 2, (4, 32, 32)).astype(np.float32)
    out = hist_match(tgt, tpl)
    np.testing.assert_allclose(np.percentile(out, QS)[5:-5], tpl[5:-5], atol=0.02)


def test_input_corrections_keep_shape():
    from cars_tta.candidates import INPUT_CANDS, correct
    v = np.random.rand(3, 16, 16).astype(np.float32)
    for n in INPUT_CANDS:
        o = correct(v, n, template=np.linspace(0, 1, 101))
        assert o.shape == v.shape and np.isfinite(o).all()


def test_candidate_features_identity_and_change():
    from cars_tta.features import candidate_features
    P = np.zeros((2, 4, 32, 32), np.float32); P[:, 0] = 1; P[:, 0, 8:20, 8:20] = 0; P[:, 1, 8:20, 8:20] = 1
    f = candidate_features(P, P)
    assert f["dis_delta"] == 0 and f["dis_dice"] == 1.0 and f["frag_n"] == 0
    Q = P.copy(); Q[:, 1, 8:20, 8:12] = 0; Q[:, 3, 8:20, 8:12] = 1
    g = candidate_features(P, Q)
    assert g["dis_delta"] > 0 and g["frag_n"] >= 1 and g["dis_dice"] < 1


def test_wide_router_shapes_and_safe_default():
    import pandas as pd
    from cars_tta.router import WideRouter, choose_tau_wide, wide_arrays
    rng = np.random.default_rng(0)
    n, cands = 120, ["hi1", "g07"]
    df = {"fs_unc_ent0": rng.random(n), "q__src": np.full(n, 0.8)}
    for c in cands:
        x = rng.normal(size=n)
        df[f"fc__{c}__dis_delta"] = x
        df[f"q__{c}"] = 0.8 + 0.02 * x  # gain is a function of the feature
    df = pd.DataFrame(df)
    S_, C_, G, _, _ = wide_arrays(df, cands)
    assert S_.shape == (n, 1) and C_.shape == (n, 2, 1) and G.shape == (n, 2)
    r = WideRouter(cands, epochs=60, n_ens=1).fit(S_, C_, G)
    choose_tau_wide(r, S_, C_, G)
    d = r.decide(S_, C_)
    gfull = np.concatenate([np.zeros((n, 1)), G], 1)
    assert gfull[np.arange(n), d].mean() > 0.005  # learns to pick helpful candidates
    assert r.decide(S_, C_, tau=10.0).max() == 0  # a huge margin always keeps the source


def test_consensus_features():
    from cars_tta.features import candidate_features, consensus_spread
    P = np.zeros((2, 4, 16, 16), np.float32); P[:, 0] = 1; P[:, 0, 4:10, 4:10] = 0; P[:, 1, 4:10, 4:10] = 1
    f = candidate_features(P, P, Pc=P)
    assert f["dis_cons_dice"] == 1.0 and f["dis_cons_l1"] == 0.0
    assert consensus_spread([P, P], P)["dis_spread"] == 0.0
