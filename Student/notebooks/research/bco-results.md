# BCO Results

## Baselines

| Method | Reward |
|---|---|
| Teacher (TD3) | 261 +/- 26 |
| Random policy | -234 +/- 20 (avg over 3 runs) |

## Latent-based BCO

BC policy takes autoencoder latents as input. IDM trained on random rollouts, pseudo-labels expert frame transitions, BC trained on pseudo-labels. All results averaged over 3 runs.

### Single frame

| Frames | Stride | Reward (mean) | Test MSE |
|---|---|---|---|
| 1 | - | -283.6 | 0.122 |

### Frame stacking

Stacking multiple latents to capture motion with a stride being used for skipping frames.

| Frames | Stride | Window (frames) | Reward (mean) | Test MSE |
|---|---|---|---|---|
| 4 | 1 | 3 | -268.5 | 0.114 |
| 4 | 8 | 24 | -257.8 | 0.109 |
| 4 | 16 | 48 | -159.4 | 0.108 |
| 4 | 32 | 96 | -129.5 | 0.102 |
| 4 | 64 | 192 | -128.4 | 0.083 |
| 8 | 16 | 112 | -134.2 | 0.099 |
| 8 | 32 | 224 | -137.0 | 0.056 |

Best: 4 frames stride 64 at -128.4 with wider strides helping since consecutive frames are too similar for the AE to preserve the differences. Plateaus around -130 regardless of more frames or wider strides.

## Takeaway

The autoencoder latent is the bottleneck. It's trained for reconstruction so it keeps visual info but drops dynamics (velocity, angular rate). Frame stacking partially recovers motion info and gets the reward above random (-234), but the gap to the Teacher (261) remains large.
