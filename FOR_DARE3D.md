# Contexte pour implémenter le plugin napari DARE3D

Ce document résume **ce qui a été fait pour le plugin napari DARE2D** (`napari-dare2d/`),
pour servir de base à l'équivalent **DARE3D** (même logique, en 3D). DARE3D :
https://github.com/JFRupprecht-OM/DARE3d

> Lis aussi `HANDOFF.md` (détail complet, daté) — ici c'est la version condensée + ce qui change en 3D.

---

## 1. Objectif (identique en 3D)
Un plugin napari qui : (1) lance l'inférence DARE (TensorFlow) sur une séquence chargée
dans napari, (2) superpose les résultats (centres de division, orientation, longueur d'axe)
en layers napari. Le code DARE n'utilise pas napari nativement : ce sont des **CLI**
(click pour l'inférence, argparse pour le post-traitement) orchestrées par un notebook via
`subprocess`. On a écrit une **fine couche API in-process** par-dessus, puis un widget.

## 2. Environnement (le plus transférable — mêmes contraintes attendues en 3D)
- Env conda **`napari-env-for-DARE2D-claude`**, Python 3.10, **CPU** (TF 2.12 n'a pas de GPU
  sur Windows natif ; GPU = WSL2/Linux, sans changement de code).
- **numpy 1.23.5 est le pivot** : TF 2.12 impose `numpy<1.24`. Toute la stack est donc figée
  à la génération 2023 : **napari 0.4.18** (la dernière qui tolère numpy 1.23 ; napari ≥0.7
  exige numpy≥2 → conflit), scikit-image 0.21, opencv 4.11, pandas 2.0, numba 0.57, pydantic 1.10.
- **Piège n°1** : ne jamais « upgrader » napari/numpy → ça retire le pin et casse TF. À l'install
  du plugin : `pip install --no-build-isolation --no-deps -e .` et **`dependencies = []`** dans
  le `pyproject.toml` (sinon pip re-résout et tire numpy≥2).
- **Piège n°2** : `segmentation_models` ne s'importe qu'avec `os.environ['SM_FRAMEWORK']='tf.keras'`
  **avant** l'import (DARE le fait déjà dans ses modules modèle). Requis pour bâtir l'archi U-Net.
- Réseau d'entreprise → install pip avec `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

## 3. L'API in-process (`napari_dare2d/_api.py`) — patron à reproduire en 3D
Numpy pur, **n'importe pas napari** (donc testable/headless). Réutilise le code DARE existant
au lieu de le réécrire. Fonctions clés :
- `build_models(reg_ckpt, seg_ckpt)` : instancie les modèles via **Hydra**. Détail critique :
  les scripts font `hydra.initialize(config_path="../../config")` (relatif au fichier → casse
  depuis un plugin). On utilise **`initialize_config_dir(config_dir=<abs>)`** + on **clear le
  singleton** `GlobalHydra.instance().clear()` autour de chaque build (ré-entrance dans une
  session napari longue). Les `.h5` sont des **poids seuls** (`load_weights`, pas `load_model`).
- `infer_stack(stack, reg, seg, frames, progress_cb)` : refactor de la boucle d'inférence du
  script, **sans I/O ni dessin** → `{frame: [ {x,y,angle,length} ]}`.
- `run_ensemble(...)` : boucle les 8 sets de modèles, `K.clear_session()` entre chaque.
- `consensus(all_dets, n_frames, eps, min_models, num_models)` : clustering (DBSCAN, hdbscan
  absent) + agrégation médiane → consensus par frame. Réutilise les primitives du post-traitement.
- Helpers : `parse_sets("1-8")`, `find_checkpoints` (layout `checkpoints_set_{n}_all_but_target/best.h5`),
  `resolve_frames`.
- **Piège n°3** : `scripts/` n'a pas de `__init__.py` → importable seulement comme *namespace
  package* après avoir mis `DARE2d-main/` sur `sys.path` (le shim le fait).

## 4. Mapping vers les layers napari (`_api.to_layer_data`) — **ce qui change le plus en 3D**
- **Convention de coordonnées (verrouillée en 2D)** : détection `x = colonne`, `y = ligne`.
  Un point napari sur un stack `(T,Y,X)` est donc `(t, y, x)`. (Vérifié par comparaison à une
  référence : une inversion x/y faisait chuter l'erreur médiane de ~350 px à ~18 px.)
- **Points** : `(N,3)` = `(t, y, x)`, + `properties` (angle, length, et stats consensus).
- **Vectors** : `(N,2,3)` ; `v[:,0]`=origine à une extrémité, `v[:,1]`=direction
  `(0, cos(angle)·L, sin(angle)·L)` (cohérent avec `project_point` du script). Avec `length=1`
  le segment est centré sur la détection.
- **Défauts du layer Vectors** (demandés) : `vector_style="line"`, `edge_width=5` ;
  Points `size=24`. napari 0.4.18 supporte bien `vector_style` (line/triangle/arrow).
- `to_layer_data(per_frame, frame_base=...)` renvoie une liste de `LayerDataTuple`. **`frame_base`** :
  `0` pour `infer_stack` (clés 0-based), `1` pour `consensus` (clés 1-based). napari t = clé − frame_base.

## 5. Le plugin npe2 (`napari-dare2d/`)
```
pyproject.toml            # dependencies = []  (volontaire — protège le pin numpy)
napari_dare2d/
  __init__.py             # léger : n'importe PAS _widget (évite de tirer Qt avec _api)
  napari.yaml             # manifeste npe2 : commands + widgets
  _api.py                 # l'API ci-dessus
  _widget.py              # widget magic_factory
verify_api.py             # check réel (modèles + données)
verify_layers.py          # check rapide (géométrie + acceptation napari, sans modèles)
```
- Manifeste : `contributions.commands` (`id`, `title`, `python_name: napari_dare2d._widget:dare2d_widget`)
  + `contributions.widgets` (`command`, `display_name`). Entry point `napari.manifest` dans pyproject.
  Notre widget est un **`@magic_factory`** (pas `autogenerate`, qui ne sert qu'aux fonctions nues).
- **Widget** : entrées = layer Image, dossiers checkpoints (défauts résolus via `__file__`),
  `model_sets` ("1-8"/"8"), plage de frames, `eps`/`min_models`, barre de progression. Tourne dans un
  `napari.qt.threading.thread_worker` **générateur** qui *yield* la progression (par frame en mono-modèle,
  par modèle en ensemble). 1 set → détections brutes ; ≥2 → consensus. Ajoute via `viewer._add_layer_from_data`.
- **Pièges napari 0.4.18** : Points utilise **`edge_color`** (renommé `border_color` en ≥0.5) ;
  les **captures d'écran ne marchent qu'en fenêtre visible** (en `QT_QPA_PLATFORM=offscreen`, pas de
  contexte GL → `viewer.screenshot()` lève). Construction du widget et création des layers OK en headless.

## 6. Validation faite (méthodo réutilisable)
- `verify_api.py` : build set 8 réel + `infer_stack` sur quelques frames d'un vrai stack
  `(56,1024,1024) uint8`, asserte structure/types/plages, compare aux `.npy` de référence
  (anciens, format `[x,y,2]` → ancre lâche), teste `consensus`. **OK**.
- `verify_layers.py` : géométrie (angle→vecteur, segment centré, |segment|=L), conventions,
  acceptation headless par napari. **OK**.
- Test réel : widget piloté dans napari (set 8, 3 frames) → layers créés en ~9 s ; justesse
  confirmée par un rendu matplotlib (centres sur les cellules, axes centrés).
- `infer_stack` exige du **8-bit** (lève sur uint16) — un rescale 16-bit est un follow-up.

---

## 7. Ce qui change pour DARE3D (points d'attention)
- **Dimensions** : stacks `(T, Z, Y, X)` (ou `(Z,Y,X)` sans temps). Adapter :
  - Points napari → `(t, z, y, x)` (`(N,4)`), Vectors → `(N,2,4)`.
  - Fenêtre glissante / patches en **3D** (volumes), pas en 2D ; le coût CPU explose →
    prévoir progression fine, sous-échantillonnage Z, et tester d'abord sur petit volume.
- **Orientation 3D** : en 2D c'est un angle scalaire (`cos2θ/sin2θ` → degrés). En 3D, DARE2D a
  déjà des configs `config/angle_representation_3d/` : **quaternion**, **rotation_6d**, **rotation_9d_svd**.
  Le mapping `to_vectors` devra convertir cette représentation en **vecteur unité 3D** `(dz,dy,dx)`
  pour le layer Vectors (axe de division dans l'espace). C'est le vrai morceau nouveau.
- **Modèles** : segmentation/regression 3D (archi et `target_size`/`crop_size` 3D) — vérifier le
  `_target_` Hydra des modèles 3D dans le repo DARE3D, et si `segmentation_models` a un équivalent 3D
  (souvent non → archi U-Net 3D maison) ; sinon adapter `build_models`.
- **Consensus** : clustering spatial en 3D (DBSCAN sur `(z,y,x)`), agrégation d'orientation = moyenne
  de quaternions/rotations (pas une simple médiane d'angle). Réutiliser la structure 2D mais changer
  la métrique d'angle.
- **Constant** : tout le reste du patron tient — env figé numpy 1.23/TF 2.12, API in-process napari-free,
  Hydra `initialize_config_dir`+clear, `SM_FRAMEWORK`, npe2 + magic_factory + thread_worker générateur,
  `dependencies=[]`, conventions de coordonnées (row/col/plane), checks `verify_*`.
- **Réutilisation** : si DARE3D et DARE2D partagent l'archi du code (Hydra/scripts CLI), le shim 3D peut
  être quasi un copier-adapter de `_api.py` : changer la dimensionnalité des tableaux, la fenêtre
  glissante, et la conversion d'orientation. Garder `_api.py` napari-free + un `verify_layers.py` rapide.
