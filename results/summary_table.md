| Model | Trainable params | Best val acc | Test accuracy | Macro F1 |
|---|---|---|---|---|
| Baseline (CNN from scratch) | 259,338 | 0.6879 | 0.6911 | 0.6351 |
| Ablation A -- frozen ResNet18 backbone | 5,130 | 0.5716 | 0.5548 | 0.5157 |
| Ablation B / deployed -- fine-tuned ResNet18 | 11,181,642 | 0.9011 | 0.8866 | 0.8757 |

### Per-class performance (deployed model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bacterial_leaf_blight | 0.842 | 0.877 | 0.859 | 73 |
| bacterial_leaf_streak | 0.862 | 0.983 | 0.918 | 57 |
| bacterial_panicle_blight | 1.000 | 0.712 | 0.832 | 52 |
| blast | 0.937 | 0.794 | 0.860 | 262 |
| brown_spot | 0.886 | 0.856 | 0.871 | 146 |
| dead_heart | 0.936 | 0.945 | 0.940 | 217 |
| downy_mildew | 0.820 | 0.785 | 0.802 | 93 |
| hispa | 0.879 | 0.942 | 0.909 | 240 |
| normal | 0.868 | 0.962 | 0.913 | 266 |
| tungro | 0.840 | 0.866 | 0.853 | 164 |
