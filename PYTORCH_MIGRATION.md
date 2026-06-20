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
