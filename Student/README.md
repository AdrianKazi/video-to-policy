# Student

`src/` contains the modular code.
`notebooks/research/v3...` contains the research workflow.

Both the modular code and the research notebooks write all datasets, models, plots, and other artifacts to `Student/runs/...`.

## Environment

Before running the notebook or the modular pipeline:

```bash
cd Student
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Teacher Videos and Frames

Source videos live in:

```text
Teacher/videos/
```

To generate raw frames for Student:

```bash
cd Student
source .venv/bin/activate
export PYTHONPATH=src
python src/teacher_data_prep.py
```

This writes frames to:

```text
Student/data/frames/
```

## Autoencoder

Input:

- frame: `(1, 84, 84)`
- batch: `(B, 1, 84, 84)`

Output:

- latent: `(64,)` or `(B, 64)`
- reconstruction: `(1, 84, 84)` or `(B, 1, 84, 84)`

Artifacts go to:

```text
Student/runs/autoencoder/autoencoder_<stamp>/
```

Typical flow:

```bash
python -m Autoencoder dataset
python -m Autoencoder train
python -m Autoencoder eval
```

## Sequential

Input:

- latent sequence: `(T, 64)`
- batch of sequences: `(B, T, 64)`

Output:

- predicted next latent: `(64,)` or `(B, 64)`
- decoded next frame: `(1, 84, 84)` or `(B, 1, 84, 84)`

Artifacts go to:

```text
Student/runs/sequential/sequential_<stamp>/
```

Typical flow:

```bash
python -m Sequential dataset
python -m Sequential train
python -m Sequential eval
```
