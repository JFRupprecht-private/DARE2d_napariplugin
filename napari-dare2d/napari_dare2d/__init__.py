"""napari-dare2d: run DARE2D cell-division detection and overlay results in napari.

Kept import-light on purpose: the widget (which pulls in Qt/napari) lives in
``napari_dare2d._widget`` and is loaded by npe2 via the napari.yaml manifest,
so importing this package (e.g. for the headless ``_api``) does not drag in Qt.
"""

import os

# TensorFlow (Keras backend) and PyTorch (the optional torch backend) can both be
# loaded in one process; on Windows they each ship an Intel OpenMP runtime, whose
# duplicate-load check aborts the process ("OMP: Error #15"). Allow the duplicate
# load -- must be set before either lib initialises OpenMP, hence here at import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
