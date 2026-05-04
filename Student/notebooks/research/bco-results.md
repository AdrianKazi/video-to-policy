# BCO Results

## Baselines

| Method | Reward |
|---|---|
| Teacher (TD3) | 261 +/- 26 |
| Random policy | -205 +/- 141 |

## Latent-based BCO

BC policy takes autoencoder latents as input. IDM trained on random rollouts, pseudo-labels expert frame transitions, BC trained on pseudo-labels.

### Single frame

| Frames | Stride | Reward | Test MSE |
|---|---|---|---|
| 1 | - | -293 | 0.194 |

### Frame stacking

Stacking multiple latents to capture motion with a stride being used for skipping frames.

| Frames | Stride | Window (frames) | Reward | Test MSE |
|---|---|---|---|---|
| 4 | 1 | 3 | -240 | 0.114 |
| 4 | 8 | 24 | -158 | 0.108 |
| 4 | 16 | 48 | -144 | 0.108 |
| 4 | 32 | 96 | -123 | 0.099 |
| 4 | 64 | 192 | -125 | 0.079 |
| 8 | 16 | 112 | -135 | 0.095 |
| 8 | 32 | 224 | -136 | 0.052 |

Best: 4 frames stride 32 at -123 with wider strides helping since consecutive frames are too similar for the AE to preserve the differences.