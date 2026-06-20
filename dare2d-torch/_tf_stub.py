"""Install a minimal fake ``tensorflow`` so DARE2D's inference utils import TF-free.

DARE2d-main/dare2d/evaluation/center_metrics.py does ``import tensorflow as tf``
only to declare Keras *metric* classes (``class X(tf.keras.metrics.Metric)``) that
the inference path never touches -- but ``napari_dare2d._api`` imports
``extract_centers`` from that module, so the import drags TF in. TF can't coexist
with numpy 2 / onnxruntime here, so we shim the tiny surface used *at import time*
(the metric base class). Real TF is never needed for ONNX inference.

ponytail: stub satisfies only ``tf.keras.metrics.Metric`` (base class) and
``tf.VariableAggregation`` (used inside metric __init__, not on our path).
Ceiling: if DARE2d-main later uses more of ``tf`` at module/inference scope, this
stub must grow -- but the right fix then is a real torch/onnx port of that util.

Import this BEFORE ``napari_dare2d._api``.
"""

import sys
import types


def install():
    if "tensorflow" in sys.modules:
        return
    tf = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    metrics = types.ModuleType("tensorflow.keras.metrics")

    class _Metric:  # subclassable placeholder for tf.keras.metrics.Metric
        pass

    metrics.Metric = _Metric
    keras.metrics = metrics
    tf.keras = keras

    class _VarAgg:
        ONLY_FIRST_REPLICA = 0

    tf.VariableAggregation = _VarAgg

    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.keras"] = keras
    sys.modules["tensorflow.keras.metrics"] = metrics


install()
