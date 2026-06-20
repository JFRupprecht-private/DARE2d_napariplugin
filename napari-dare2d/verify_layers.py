"""Runnable check for the napari layer mapping (_dare2d_api layer functions).

Pure geometry + a headless napari acceptance test. Fast (no models / no TF
inference). Asserts:
  - point coords are (t, y, x) with the locked x=col / y=row convention,
  - vector geometry matches the script's project_point (angle 0 -> +row axis,
    rod centred on the detection, |rod| == length),
  - consensus 1-based keys map via frame_base=1,
  - napari 0.4.18 actually accepts the produced LayerDataTuples.

Run:  <env>/python.exe napari-dare2d/verify_layers.py   (exit 0 = OK)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from napari_dare2d import _api as api  # noqa: E402

failures = []


def check(cond, msg):
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if not cond:
        failures.append(msg)


def main():
    # frame 0: one detection at col=100,row=200, angle 0deg, length 40
    # frame 2: two detections
    per_frame = {
        0: [{"x": 100, "y": 200, "angle": 0.0, "length": 40.0}],
        2: [
            {"x": 300, "y": 50, "angle": 90.0, "length": 20.0},
            {"x": 10, "y": 10, "angle": 45.0, "length": 10.0},
        ],
    }

    print("== points ==")
    pts, props = api.detections_to_points(per_frame)
    check(pts.shape == (3, 3), f"points shape (3,3): {pts.shape}")
    # first point: (t=0, y=200, x=100)
    check(np.allclose(pts[0], [0, 200, 100]), f"point0 == (t,y,x)=(0,200,100): {pts[0].tolist()}")
    check(np.allclose(sorted(props["angle"]), [0, 45, 90]), "angle property carried")
    check(np.allclose(sorted(props["length"]), [10, 20, 40]), "length property carried")

    print("== vectors ==")
    vecs = api.detections_to_vectors(per_frame)
    check(vecs.shape == (3, 2, 3), f"vectors shape (3,2,3): {vecs.shape}")
    # detection 0: angle 0 -> direction (0, cos0*L, sin0*L) = (0, 40, 0) (along +row)
    o0, d0 = vecs[0]
    check(np.allclose(d0, [0, 40, 0]), f"angle0 direction == (0,L,0): {d0.tolist()}")
    # rod centred on the detection point (t,y,x)=(0,200,100)
    mid = o0 + d0 / 2.0
    check(np.allclose(mid, [0, 200, 100]), f"rod midpoint == centre: {mid.tolist()}")
    # |rod| == length
    check(np.isclose(np.linalg.norm(d0), 40.0), f"|rod| == length(40): {np.linalg.norm(d0):.3f}")
    # angle 90 -> direction (0, ~0, L) along +col
    d_90 = vecs[1, 1]
    check(np.allclose(d_90, [0, 0, 20], atol=1e-6), f"angle90 direction == (0,0,L): {d_90.tolist()}")

    print("== consensus frame_base=1 ==")
    cons_like = {1: [{"x": 5, "y": 6, "angle": 0.0, "length": 8.0,
                      "angle_std_deg": 1.0, "n_models": 3}]}
    cpts, cprops = api.detections_to_points(cons_like, frame_base=1)
    check(np.allclose(cpts[0], [0, 6, 5]), f"1-based key -> t=0: {cpts[0].tolist()}")
    check("angle_std_deg" in cprops and "n_models" in cprops, "consensus extra props carried")

    print("== empty input ==")
    epts, _ = api.detections_to_points({})
    evecs = api.detections_to_vectors({})
    check(epts.shape == (0, 3) and evecs.shape == (0, 2, 3), "empty -> (0,3) / (0,2,3)")

    print("== napari 0.4.18 accepts the LayerDataTuples (headless) ==")
    try:
        import napari
        v = napari.Viewer(show=False)
        for data, meta, ltype in api.to_layer_data(per_frame):
            v._add_layer_from_data(data, meta, ltype)
        names = [ly.name for ly in v.layers]
        axes = next(ly for ly in v.layers if "axes" in ly.name)
        style = str(axes.vector_style).split(".")[-1].lower()
        width = axes.edge_width
        v.close()
        check(any("centers" in n for n in names) and any("axes" in n for n in names),
              f"Points + Vectors layers created: {names}")
        check(style == "line" and width == 5,
              f"axes default: vector_style={style} edge_width={width}")
    except Exception as e:
        check(False, f"napari acceptance raised {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
        return 1
    print("RESULT: OK — detections/consensus map cleanly to napari Points + Vectors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
