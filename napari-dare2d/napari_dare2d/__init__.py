"""napari-dare2d: run DARE2D cell-division detection and overlay results in napari.

Kept import-light on purpose: the widget (which pulls in Qt/napari) lives in
``napari_dare2d._widget`` and is loaded by npe2 via the napari.yaml manifest,
so importing this package (e.g. for the headless ``_api``) does not drag in Qt.
"""

__version__ = "0.1.0"
