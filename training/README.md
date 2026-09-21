# Training pipeline notes

## Pretrained weights

`torchvision`'s normal path for ImageNet-pretrained ResNet18 weights downloads from
`download.pytorch.org`, which is not reachable from a network-restricted environment. This
project instead ships `pretrained/resnet18-5c106cde.pth`, an ImageNet-pretrained ResNet18
checkpoint mirrored in the public GitHub repo
[fregu856/deeplabv3](https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/resnet/resnet18-5c106cde.pth)
(an older torchvision release's checkpoint, from before the hash in the filename changed
naming scheme). Before relying on it we verified:

- `state_dict` keys match `torchvision.models.resnet18()`'s keys exactly (`load_state_dict(..., strict=True)` succeeds with no missing/unexpected keys).
- Weight norms are in a sane range (not zero, not exploded).
- A forward pass on random input produces a valid 1000-way ImageNet logit vector with no NaNs.

See `models.py` for how it's loaded.

## Pipeline

1. `make_splits.py` -- builds `data/manifest.csv`, a seeded stratified 70/15/15
   train/val/test split over `data/paddy_resized_160/<class>/*.jpg`.
2. `dataset.py` -- `RiceDiseaseDataset`, a thin `torch.utils.data.Dataset` over the manifest.
3. `models.py` -- `build_model(mode, n_classes)` for `"baseline"` / `"frozen"` / `"finetuned"`.
4. `train.py` -- trains one mode, checkpoints the best-val-accuracy model to
   `models/<mode>_best.pt`, and writes `results/<mode>_history.csv` and
   `results/<mode>_test_eval.json` (test accuracy, macro F1, per-class precision/recall/F1,
   confusion matrix).
5. `run_all.sh` -- runs all three modes back to back (baseline, frozen, finetuned) and
   touches `results/ALL_DONE` when finished.

Run a single experiment directly, e.g.:

```bash
python3 train.py --mode finetuned --epochs 10
```
