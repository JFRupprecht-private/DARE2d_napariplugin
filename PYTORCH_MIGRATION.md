# Plan de migration DARE2D : TensorFlow/Keras → PyTorch

_Mode « Ultracode » — plan rigoureux et phasé. Daté 2026-06-18._
_À lire avec `HANDOFF.md` (état actuel TF) et `FOR_DARE3D.md`._

---

## DÉCISIONS PRISES (2026-06-18) — plan bi-pistes, inférence seule
- **Objectif = « les deux »** → **Piste 1 d'abord : ONNX pour le GPU tout de suite** (stratégie C,
  risque faible, poids exacts), **puis Piste 2 en fond : portage PyTorch progressif** (stratégie A).
- **Périmètre = inférence seule** → on porte uniquement le chemin du plugin
  (`build_models` + `infer_stack` + les 2 modèles). **L'entraînement est HORS périmètre**
  (trainer/losses/datamodule/métriques `tf.keras` : on n'y touche pas). Stratégie B (réentraînement)
  écartée (pas de données locales, et hors périmètre).
- Conséquence : le découpage de référence est le **§9-bis** ci-dessous (les deux pistes), pas le §9 d'origine.

---

## ÉTAT D'AVANCEMENT — PISTE 1 (ONNX/GPU) ✅ TERMINÉE (2026-06-20)
Tout vit dans **`dare2d-torch/`** ; `DARE2d-main/` et `napari-dare2d` **non modifiés**.
- **P1.0** ✅ Outillage export (`onnx 1.15`, `onnxruntime 1.16.3`, `tf2onnx 1.16.1 --no-deps`)
  ajouté à l'env TF sans bouger numpy 1.23.5 / protobuf 4.25.9. Env GPU `dare2d-onnx`
  (py3.11, numpy 2) : **CUDAExecutionProvider actif sur la RTX 5000** (asserté).
- **P1.1** ✅ Export des **8 reg + 8 seg** `.h5 → .onnx` (batch dynamique, opset 17) via le
  chemin Hydra de DARE2D (archi identique garantie).
- **P1.2** ✅ `backends.OnnxModel` = drop-in Keras (`.model.predict`) → `infer_stack` /
  `inference_strategy` réutilisés **verbatim**.
- **P1.4** ✅ **Parité TF↔ONNX** (sets 1/5/8) : seg `max|Δ|≈7e-7`, reg `≈1e-7`, masque 100 %,
  **détections identiques** (comptes, centres 0 px, angle/longueur 0).
- **P1.5** ✅ **GPU** : seg ~29×, reg ~16×, **end-to-end réel ~18,6×** (2856→154 ms/frame),
  détections GPU == CPU. **Livrable : GPU exploitable, poids exacts.**
- **Pièges rencontrés & corrigés** : ort-gpu 1.27 exige CUDA 13 (stub PyPI) → 1.22 (CUDA 12) ;
  cuDNN charge ses sous-libs via **PATH** (pas `add_dll_directory`) ; `PYTHONNOUSERSITE` (numpy
  user-site fuite dans l'env) ; **numpy 2** refuse `values[k,0]=array(1,)` dans `convert_values`
  → backend renvoie `length` en 1-D (B,) = shape documentée ; encodage console cp1252 (Δ).

## ÉTAT D'AVANCEMENT — PISTE 2 (PORTAGE PYTORCH) ✅ RÉGRESSION ; SEG = REPLI ONNX (2026-06-20)
- **P2.0** ✅ Env `dare2d-torch` (py3.11, numpy 2) : **torch 2.6+cu124, CUDA voit la RTX 5000**,
  `smp.Unet('resnet18')` se construit. Oracle = `onnxruntime` CPU dans le même env (les `.onnx`
  Piste 1 sont fidèles à TF à 1e-7 → oracle portable, pas de pont TF).
- **P2.1** ✅ **Régression portée fidèlement** : `Regression2dTorch` + conversion de poids
  (`keras_dump.py`→`convert_to_torch.py` : Conv HWIO→OIHW, Dense transpose, **piège Flatten NHWC
  corrigé** via `permute(0,2,3,1)`). **Parité torch↔ONNX `max|Δ|≈1e-7` sur les 8 sets**. Asserts de
  shapes à la conversion (garde-fou archi).
- **P2.2** ✅ **U-Net porté fidèlement en torch** (+ repli ONNX conservé par défaut). `smp.Unet` n'est
  PAS compatible poids (BN `bn_data`, raccourci 1×1 `stage1_unit1_sc`, ordre **préactivation**), donc
  `SegmentationUnetTorch` est **écrit à la main d'après le graphe Keras introspecté** (`seg_arch.json`).
  Détails fidèles : resnet18 **préactivation** (BN→ReLU→conv, raccourci pris APRÈS la 1ʳᵉ BN-ReLU) ;
  qubvel utilise **ZeroPadding symétrique + conv `valid`** (PAS de `same` asymétrique !) → exact via
  `F.pad` ; **pad-zéro avant maxpool** (pas le -inf de torch) ; **deux eps BN** (encodeur 2e-5, décodeur
  1e-3) ; `bn_data` scale=False (γ=1). **Parité vs ONNX : `max|Δ|≈1.3e-7`, masque 100 % sur les 8 sets.**
- **P2.3** ✅ **Backends torch + commutateur seg** : `torch_backend.py` (`.model.predict` compatible
  Keras pour reg ET seg) → `api.infer_stack` réutilisé verbatim. `build_hybrid_models(seg_backend=…)` :
  `"onnx"` (défaut, éprouvé) ou `"torch"` (100 % torch). **End-to-end vs oracle, les DEUX backends :
  centres identiques, angle/longueur Δ≈1e-6** (`verify_torch_e2e.py --seg-backend onnx|torch`).
- **B (commutateur dans le plugin napari)** ✅ Sélecteur **Keras / PyTorch** dans le widget. Contrainte
  clé : Keras (TF 2.12) exige numpy<1.24 → ne tourne QUE dans l'env napari-0.4.18 ; **torch+cu124 ajouté
  à CET env** (numpy 1.23.5 intact, vérifié), donc bascule live Keras↔PyTorch (GPU) dans une seule
  session. `to_layer_data` rendu **compatible napari 0.4.18 ET ≥0.5** (`edge_color`/`border_color` via
  `importlib.metadata`). Coexistence TF+torch : `KMP_DUPLICATE_LIB_OK=TRUE` (clash OpenMP Windows).
  Validé : `verify_layers` vert (0.4.18), widget instancié, **détections keras == pytorch** (GPU).
- **Pièges Piste 2/B** : disque plein (cache pip 12 Go purgé) ; `PYTHONNOUSERSITE` ; torch-cuda +
  `onnxruntime` CPU cohabitent (pas de clash DLL) ; `weights_pt/` gitignoré ; eps BN par section ;
  TF+torch même process → OpenMP (`KMP_DUPLICATE_LIB_OK`).

**Livrable Piste 2** : régression ET segmentation = vrais PyTorch fidèles (parité ~1e-7). La seg reste
un **commutateur** (ONNX par défaut = mitigation du risque, torch en option) ; pipeline 100 % torch
disponible sur GPU, détections identiques à TF.

---

## 0. Pourquoi migrer (le vrai moteur)
1. **GPU sur Windows natif.** La machine a une **Quadro RTX 5000 (16 Go)** inutilisée. TF 2.12
   n'a pas de GPU sur Windows natif (≥2.11 → WSL2 obligatoire). **PyTorch a le GPU CUDA en natif
   sur Windows** → on exploite la carte sans WSL2.
2. **Sortir du carcan de versions.** TF 2.12 impose `numpy<1.24`, ce qui force napari **0.4.18**
   (vieux). PyTorch n'impose pas ce pin → on pourra passer à **numpy 2 + napari ≥0.5** (et corriger
   les contournements 0.4.18 : `edge_color`→`border_color`, etc.).

> ⚠️ Si l'objectif **réel** est « juste le GPU », il existe une voie bien moins risquée que PyTorch :
> **ONNX** (voir §2, stratégie C). À trancher avant de se lancer.

---

## 1. Périmètre réel (ce qui doit changer vs ce qui ne bouge pas)
**Coté inférence / plugin — surface minuscule :**
- `dare2d/model/regression2d_cnn.py` : `Regression2dCNN` (Keras) → `nn.Module`.
- `dare2d/model/segmentation2d_cnn.py` : `Segmentation2dCNN` = `sm.Unet(resnet18)` → `smp.Unet`.
- `napari_dare2d/_api.py` : `build_models` (charger des poids torch) + les 2 appels modèle
  (`seg_model.model.predict`, `reg_model.model.predict`) → tenseurs torch.
- **Inchangé** (numpy/cv2/skimage) : `inference_strategy` (fenêtre glissante), `extract_centers`,
  `crop_img_from_center`, `convert_values`, tout `to_layer_data`, le widget, le consensus.

**Hors périmètre plugin** (seulement si on réentraîne) : les 22 fichiers TF restants
(training_pipeline, trainer, callbacks, losses, datamodule, métriques `tf.keras.metrics`).
À NE PAS porter pour faire tourner l'inférence.

---

## 2. Le point dur : les poids entraînés (DÉCISION À PRENDRE)
Il y a **8 checkpoints régression + 8 segmentation** (`.h5`, poids Keras), et **pas de données
d'entraînement sur le disque**. Trois stratégies :

| | Stratégie | Réutilise les poids ? | Risque | GPU Windows natif |
|---|---|---|---|---|
| **A** | **Conversion de poids** Keras→PyTorch (reconstruire l'archi en torch + transférer les tenseurs) | Oui | Régression : faible. **U-Net : élevé** | Oui |
| **B** | **Réentraînement** en PyTorch (reproduire le pipeline d'entraînement) | Non (refait) | Élevé + **bloqué : pas de données locales** | Oui |
| **C** | **Pont ONNX** : `tf2onnx` exporte les `.h5` → `onnxruntime-gpu` pour l'inférence | Oui (exacts) | **Faible** (pas de réécriture de modèle) | **Oui** |

- **A** est le « vrai » PyTorch demandé. La **régression** se convertit mécaniquement
  (Conv2D + Dense). Le **U-Net resnet18** est le point sensible : `segmentation_models` (TF) et
  `segmentation_models_pytorch` (smp) sont du même auteur (qubvel) mais **les décodeurs/noms de
  couches diffèrent** → le transfert de poids n'est pas 1:1 et doit être validé couche par couche.
- **C** atteint l'objectif GPU **sans réimplémenter les modèles** ni risque de divergence
  (les poids restent exacts), mais ce n'est pas « PyTorch ». À considérer sérieusement si le but est
  la perf GPU et non un codebase torch.

**Recommandation :** si but = GPU → **C** (rapide, sûr). Si but = vrai codebase PyTorch → **A**,
avec la **régression d'abord** (gain facile) puis le **U-Net** en pesant le risque, fallback **C**.

---

## 3. Environnement cible (nouveau, moderne — sans les pins TF)
- Nouvel env conda, p.ex. `dare2d-torch`, **Python 3.11**.
- `torch` 2.x **+cu121** (roue CUDA Windows native), `torchvision`.
- `segmentation-models-pytorch`, `timm` (encodeurs).
- `numpy` 2, `napari` ≥0.5 `[all]`, `magicgui`, `npe2`, `scikit-image`, `opencv-python`, `scipy`,
  `tifffile`, `hydra-core`, `omegaconf`, `scikit-learn`.
- **Spike d'entrée** : `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  → doit afficher `True Quadro RTX 5000`.
- (Pour la stratégie A/validation, garder l'**ancien env TF** `napari-env-for-DARE2D-claude` comme
  oracle numérique.)

---

## 4. Portage des modèles (stratégie A)
- **Regression2dCNN → `nn.Module`** : 4× `Conv2d(k=3) → ReLU → MaxPool2d`, filtres 128→… (doublés),
  `Flatten`, puis deux têtes : `Linear(_,1)+Sigmoid` (length, `length_output`) et
  `Linear(_,2)+Tanh` (angle, `angle_output`). Attention : Keras `Conv2D` **sans padding** = `padding=0`
  (valid) ; `MaxPooling2D()` défaut = `kernel=2,stride=2`. NHWC→NCHW.
- **Segmentation2dCNN → `smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1,
  activation="sigmoid")`**. `encoder_weights=None` ⇒ pas de téléchargement (reste 100% local).
- **Hydra** : conserver la structure ; changer les `_target_` des configs `model/*.yaml` vers les
  classes torch, garder `input_channels`/`crop_size`/`backbone`.

---

## 5. Transfert de poids + validation numérique (le cœur de A)
1. **Extraire** les poids Keras dans l'ancien env TF : charger chaque `.h5`
   (`model.load_weights`), parcourir `model.layers`, sauver chaque tenseur en `.npy`/`.npz`.
2. **Mapper vers le `state_dict` torch** :
   - `Conv2D` kernel **HWIO → OIHW** (`np.transpose(k,(3,2,0,1))`), biais inchangé ;
   - `Dense` poids **(in,out) → (out,in)** (transpose), biais inchangé ;
   - `BatchNorm` : `gamma→weight, beta→bias, moving_mean→running_mean, moving_var→running_var`,
     **vérifier `eps`** (Keras 1e-3 vs torch 1e-5) ;
   - U-Net : aligner ordre des blocs encodeur/décodeur, mode d'**upsampling** (nearest/bilinear) et
     **padding** — c'est là que ça casse en silence.
3. **Gate de parité** : même entrée → comparer sorties TF vs torch, `max|Δ|` par étage sous tolérance
   (p.ex. 1e-3 après sigmoid). Faire au niveau **seg map** ET **(length, angle)**. Tant que la parité
   n'est pas atteinte sur le U-Net, ne pas avancer (ou basculer sur C).

---

## 6. Adaptation de l'inférence dans `_api.py`
- `build_models` : instancier les `nn.Module`, charger le `state_dict`, `.to(device).eval()`.
- `infer_stack` : `x (H,W,3) float32` → tenseur `(1,3,H,W)` sur `device` ; `with torch.no_grad()` ;
  retour `.cpu().numpy()`. La fenêtre glissante `inference_strategy` : empiler les patchs en un
  tenseur `(N,3,256,256)` et un seul forward (batché) — adapter la ré-écriture NHWC→NCHW.
- Le seuillage, `extract_centers`, le crop, `convert_values`, `to_layer_data` : **inchangés**.
- `device = "cuda" if torch.cuda.is_available() else "cpu"`.

---

## 7. Validation bout-en-bout (acceptation)
- Réutiliser **`verify_api.py`** mais en **oracle TF-vs-torch** : faire tourner les deux pipelines
  (ancien TF, nouveau torch) sur les **mêmes frames du set 8** et comparer **directement** les
  détections (centres, angles, longueurs) — pas l'ancienne référence `[x,y,2]`. Exiger une quasi
  égalité (centres à <1 px près, angle/longueur à epsilon). **C'est le gate de « migration sans
  régression de comportement ».**
- Garder `verify_layers.py` (géométrie napari) tel quel.

---

## 8. Effets de bord plugin & env
- napari ≥0.5 : `edge_color`→**`border_color`** sur Points ; revérifier `vector_style` (toujours OK) ;
  capture offscreen toujours sans GL.
- `pyproject.toml` : on peut **assouplir** les pins (numpy n'est plus fragile), mais garder
  `dependencies=[]` + install `--no-deps` reste prudent tant que torch/napari coexistent.
- `SM_FRAMEWORK`, `initialize_config_dir`+`GlobalHydra.clear()`, namespace package `scripts/` :
  **plus nécessaires** côté torch (sauf si on garde Hydra pour instancier les modèles → garder le
  clear du singleton).

---

## 9. Phasage (chaque phase = un check exécutable)
- **Phase 0 — Spike** : env `dare2d-torch`, `torch.cuda` voit la RTX 5000, `smp.Unet("resnet18")`
  se construit. (½ j)
- **Phase 1 — Régression** : porter + convertir le modèle de régression, **parité TF↔torch** sur des
  crops 64×64. (½ j, gain facile, dérisque la méthode)
- **Phase 2 — Segmentation (U-Net)** : porter `smp.Unet`, convertir/mapper les poids, **parité** sur
  la seg map. (le morceau risqué — plusieurs jours possibles)
- **Phase 3 — Inférence** : réécrire `build_models`/`infer_stack` en torch, **gate `verify_api`
  TF-vs-torch** vert. (1 j)
- **Phase 4 — Env moderne** : monter napari ≥0.5, corriger le plugin, `verify_layers` vert. (½ j)
- **Phase 5 — Perf GPU** : ensemble complet 8 modèles × 56 frames sur la RTX 5000, mesurer le speedup
  vs CPU. (½ j)

---

## 9-bis. Découpage retenu (bi-pistes, inférence seule) ← PLAN DE RÉFÉRENCE
> **Tout le travail des deux pistes vit dans un NOUVEAU dossier `dare2d-torch/` (voir §13).
> `DARE2d-main/` n'est jamais modifié (lecture seule via import).**

### Piste 1 — ONNX + GPU maintenant (priorité, risque faible)
But : exploiter la RTX 5000 sans réécrire les modèles ni risquer les poids.
- **P1.0** Spike env : nouvel env `dare2d-onnx` (numpy 2 OK), `pip install onnxruntime-gpu tf2onnx`
  + un env TF pour l'export. Vérif : `onnxruntime.get_device()` == `GPU` et
  `'CUDAExecutionProvider' in ort.get_available_providers()`.
- **P1.1** Export : dans l'ancien env TF, reconstruire chaque modèle (Hydra) + `load_weights`, puis
  `tf2onnx` → 1 `.onnx` par checkpoint (8 reg + 8 seg). Vérif : le `.onnx` se charge dans ORT.
- **P1.2** Wrapper d'inférence : une mini-classe `OnnxModel` exposant `.predict(x)` (même signature
  NHWC que Keras) pour que `inference_strategy`/`infer_stack` marchent **sans changement**. Backend
  sélectionnable CPU/CUDA.
- **P1.3** Brancher dans `_api.build_models` (charger `.onnx` au lieu de `.h5` selon un flag/backend).
- **P1.4** Gate de parité : `verify_api` en **oracle TF↔ONNX** sur le set 8 (centres <1 px, angle/long
  à epsilon).
- **P1.5** Perf : ensemble 8×56 frames sur GPU vs CPU → mesurer le speedup. **Livrable : GPU utilisable.**

### Piste 2 — Portage PyTorch progressif (en fond, risque croissant)
But : codebase torch propre, inférence seule. Réutilise l'oracle TF (et/ou ONNX) pour la parité.
- **P2.0** Spike : env `dare2d-torch`, `torch.cuda.is_available()` voit la RTX 5000,
  `smp.Unet("resnet18")` se construit.
- **P2.1** **Régression d'abord** : `Regression2dCNN`→`nn.Module`, conversion de poids (Conv HWIO→OIHW,
  Dense transpose), **parité** sur crops 64×64. Gain facile, dérisque la méthode.
- **P2.2** **Segmentation (U-Net)** : `smp.Unet`, mapping de poids couche par couche, **parité** sur la
  seg map. Morceau risqué ; **fallback = garder ONNX (Piste 1) pour la seg** si la parité ne tombe pas.
- **P2.3** `build_models`/`infer_stack` en torch + `device`, **gate `verify_api` TF↔torch** vert.
- **P2.4** Env moderne : napari ≥0.5, corrections plugin (`border_color`…), `verify_layers` vert.

> Les deux pistes partagent la même **gate de parité** (`verify_api` en oracle) et le même périmètre
> (inférence). On peut livrer la Piste 1 et s'arrêter là, ou enchaîner la Piste 2 sans la bloquer.

---

## 10. Risques & parades
- **Mismatch de poids U-Net (smp ≠ qubvel-TF)** → risque principal. Parade : oracle TF + diff
  couche par couche ; si irréductible, **fallback ONNX (C)** qui garde les poids exacts.
- **Différences numériques fines** (padding, BN eps, upsample) → la gate de parité les attrape.
- **Pas de données pour réentraîner** → la stratégie B reste hors-jeu tant que les données ne sont
  pas fournies. À clarifier si B est envisagé.
- **Régression silencieuse de détection** → toujours comparer torch au TF d'origine, jamais juger sur
  l'ancienne référence `[x,y,2]`.

---

## 11. Estimation d'effort (ordre de grandeur)
| Voie | Effort | Risque | Résultat |
|---|---|---|---|
| **C — ONNX + GPU** | ~0,5–1 j | faible | GPU natif, poids exacts, mais pas « torch » |
| **A — régression seule en torch** | ~0,5 j | faible | preuve de concept torch |
| **A — complet (incl. conversion U-Net + parité)** | plusieurs jours | moyen-élevé | vrai codebase torch + GPU |
| **B — réentraînement torch** | semaines + données | élevé | propre, mais bloqué localement |

---

## 12. Décisions (TRANCHÉES le 2026-06-18)
1. **But réel** : ~~GPU vs codebase torch~~ → **les deux** : ONNX/GPU d'abord (C), puis port torch (A).
2. **Stratégie poids** : **conversion / réutilisation des poids** (A pour le port, C pour ONNX).
   Réentraînement (B) **écarté** (pas de données locales).
3. **Périmètre** : **inférence seule** (plugin). Entraînement **hors périmètre**.

→ Plan d'exécution = **§9-bis** (Piste 1 ONNX puis Piste 2 PyTorch).

---

## 13. Nouveau dossier auto-portant — `DARE2d-main/` INTACT (contrainte ferme)
**Règle :** tout le portage (ONNX + torch) va dans un **nouveau dossier `dare2d-torch/`**
(frère de `napari-dare2d/`). `DARE2d-main/` est **lu** (import Hydra + classes modèle + helpers
numpy) mais **jamais édité**. Le plugin `napari-dare2d` n'est pas modifié pour le cœur du portage
(un sélecteur de backend dans le widget serait une retouche mineure *ultérieure et optionnelle*).

**Astuce d'intégration (zéro édition de l'existant) :** les backends exposent **la même interface
que le wrapper Keras** — un objet avec `.model.predict(x_nhwc, verbose=0) -> numpy` (et la régression
renvoie `(length, angle)`). Du coup `napari_dare2d._api.infer_stack` **et** `inference_strategy`
(dans `scripts/`) sont **réutilisés verbatim** : on leur passe juste des « modèles » backend au lieu
des wrappers Keras. Le préprocessing (`equalizeHist`, `/255`, fenêtre glissante, `extract_centers`,
`convert_values`) reste **partagé et inchangé** → pas de divergence de prépro possible.

**Layout proposé :**
```
dare2d-torch/                 # NOUVEAU — DARE2d-main reste intact
  README.md
  hyperparams.py              # archi LUE depuis DARE2d-main/config (n_stages=4, n_start_filters=128,
                              #   crop=64, backbone=resnet18, in_channels=3) — redéclarée ici, pas d'édition amont
  models_torch.py            # Regression2dTorch(nn.Module) + build_unet (smp.Unet resnet18)
  keras_dump.py     [env TF] # Hydra+load_weights (comme _api.build_models) -> dump poids .npz + SHAPES
  convert_to_torch.py [torch]# .npz -> state_dict torch (+ .pt) ; vérifie shapes AVANT transfert
  export_onnx.py    [env TF] # Keras -> SavedModel -> tf2onnx -> .onnx (batch dynamique, opset>=13)
  backends.py                # OnnxModel / TorchModel : interface `.model.predict` compatible Keras
  verify_parity.py           # oracle TF vs onnx/torch sur le set 8 (tolérances réalistes, cf. §14-C)
  weights_onnx/ weights_pt/  # sorties (à gitignorer)
```
- **Garde-fou archi :** `keras_dump.py` dump aussi les *shapes* de chaque couche ; `convert_to_torch.py`
  asserte la correspondance des shapes **avant** de copier (attrape tout écart d'hyperparamètres —
  p.ex. n_stages 4 vs 2 — immédiatement, pas en silence).
- Les hyperparamètres viennent du **même `compose(experiment=…)`** que le build TF, pour éviter toute
  dérive (le modèle de régression effectif est 4 stages / 128 filtres, **pas** le 2/64 du yaml modèle nu).

---

## 14. Ce qui peut casser — registre détaillé (relire AVANT de coder)

### A. Piste ONNX (risque faible, mais pièges concrets)
- **Ops resize/upsample custom** de `sm.Unet` → exporter en **opset ≥ 13** ; vérifier que `tf2onnx`
  ne laisse pas d'op non convertie.
- **Batch dynamique** : la fenêtre glissante envoie un nombre **variable** de patchs (49/frame @1024²) →
  marquer l'axe batch comme dynamique à l'export, sinon l'inférence casse sur la taille.
- **Provider CUDA silencieux** : `onnxruntime-gpu` **retombe en CPU** sans erreur si CUDA/cuDNN
  manquent ou ne matchent pas → **asserter** `'CUDAExecutionProvider' in ort.get_available_providers()`
  **et** vérifier le device réellement utilisé. Aligner les versions CUDA/cuDNN d'ORT avec le driver.
- **Layout & sorties** : `tf2onnx` préserve le NHWC ; le wrapper passe le même array. Bien récupérer
  les **noms d'E/S** et **l'ordre des 2 sorties** régression (`length` puis `angle`).

### B. Piste conversion de poids torch (risque élevé)
- **#1 piège silencieux — ordre du Flatten (régression)** : Keras aplati en **NHWC** (canal = axe le
  plus rapide), torch aplati un tenseur **NCHW** (canal = le plus lent). Sans correction, la tête Dense
  lit des features **permutées** → sorties fausses *sans erreur*. Fix : `x.permute(0,2,3,1).contiguous()`
  **avant** `flatten` côté torch (pour matcher Keras).
- **Conv** : noyau **HWIO → OIHW** (`transpose(3,2,0,1)`) ; **Dense** : `(in,out) → (out,in)` ;
  biais inchangés.
- **BatchNorm eps** : Keras défaut **1e-3** vs torch **1e-5** → forcer `eps` torch = valeur réelle du
  modèle sm (à relever), sinon biais numérique partout.
- **`.eval()` obligatoire** : BN en stats courantes + pas de dropout. Oubli = sorties fausses (piège classique).
- **U-Net resnet18 — encodeur TF ≠ torch** : le resnet18 de qubvel `classification_models` (TF) n'est
  pas identique à `torchvision`/`smp` (stem 7×7, shortcuts). Surtout, le padding **`same` de TF est
  asymétrique** au stride 2 (stem) alors que torch pad **symétrique** → **désalignement spatial**.
  Mapping nom-à-nom + diff couche par couche obligatoires. **Fallback : garder l'ONNX pour la
  segmentation** (poids exacts) et ne porter en torch que la régression.
- **Upsampling** : `nearest` des deux côtés (sinon vérifier `mode`/`align_corners`). À 256×256 les
  dimensions restent paires → les concats de skip s'alignent (OK).

### C. Transverse (les deux pistes)
- **Parité réaliste, jamais bit-exact** : conv TF (oneDNN) vs cuDNN/ORT divergent ~1e-5–1e-3, amplifiés
  par le **seuil 0.5** (peut flipper des pixels de bord → **±1-2 détections**). Gate = comparer la
  **carte de proba seg pré-seuil** (tolérance relative) **ET** les détections finales (centres à quelques
  px, comptes à ±1-2). Une exigence bit-exacte échouerait à tort.
- **Deux envs en parallèle** : la conversion exige l'**oracle TF** (ancien env) + le nouvel env
  torch/onnx ; pont par fichiers (`.npz` de poids, `.npy` d'activations). Prévoir l'orchestration.
- **Pas de données d'entraînement** → impossible de re-régler : la **seule vérité** est la parité avec TF.
  Si la parité seg est inatteignable → ONNX (poids exacts) plutôt que s'acharner.
- **Frontière d'axes** : NHWC (TF/ONNX) vs NCHW (torch) au seul appel modèle ; le wrapper convertit.
  Une erreur ici serait rattrapée par la gate de parité.
- **Détection silencieuse de régression de comportement** : toujours comparer au **TF d'origine**,
  jamais à l'ancienne référence `set_8/*.npy` (`[x,y,2]`, autre pipeline).
