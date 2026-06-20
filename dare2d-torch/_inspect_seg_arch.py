"""Dump the EXACT qubvel sm.Unet graph (layers + configs + connectivity). [TF env]

Ground truth for the hand-written torch port. Writes seg_arch.json and prints a
compact topological listing with the fields that matter for faithful replication:
BN epsilon/scale/center, ZeroPadding amounts, conv strides/use_bias, UpSampling
interpolation, Concatenate skip sources.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "napari-dare2d"))

from napari_dare2d import _api as api  # noqa: E402

KEEP = {
    "Conv2D": ["filters", "kernel_size", "strides", "padding", "use_bias", "dilation_rate"],
    "BatchNormalization": ["axis", "momentum", "epsilon", "center", "scale"],
    "ZeroPadding2D": ["padding"],
    "Activation": ["activation"],
    "MaxPooling2D": ["pool_size", "strides", "padding"],
    "UpSampling2D": ["size", "interpolation"],
    "Concatenate": ["axis"],
    "InputLayer": ["batch_input_shape"],
}


def inbound_names(entry):
    out = []
    for node in entry.get("inbound_nodes", []):
        # node is a list of [layer_name, node_idx, tensor_idx, kwargs]
        for conn in node:
            if isinstance(conn, list) and conn and isinstance(conn[0], str):
                out.append(conn[0])
    return out


def main():
    seg = api._build_model("segmentation2d", None)  # arch only, no weights needed
    model = seg.model
    cfg = model.get_config()
    layers = []
    for entry in cfg["layers"]:
        cls = entry["class_name"]
        c = entry.get("config", {})
        kept = {k: c.get(k) for k in KEEP.get(cls, [])}
        try:
            wshapes = [list(w.shape) for w in model.get_layer(entry["name"]).get_weights()]
        except Exception:
            wshapes = []
        layers.append({"name": entry["name"], "class": cls, "cfg": kept,
                       "inbound": inbound_names(entry), "wshapes": wshapes})

    out = _HERE / "seg_arch.json"
    out.write_text(json.dumps(layers, indent=1))
    print(f"{len(layers)} layers -> {out.name}")
    for L in layers:
        ws = f" w={L['wshapes']}" if L["wshapes"] else ""
        print(f"  {L['name']:24s} {L['class']:20s} {L['cfg']} <- {L['inbound']}{ws}")


if __name__ == "__main__":
    main()
