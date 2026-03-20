# 🚀 DDPG and TD3 for LunarLander

## 📌 Introduction

Reinforcement Learning (RL) is a powerful approach for solving complex control tasks, particularly in **continuous action spaces**. **Deep Deterministic Policy Gradient (DDPG)** and **Twin Delayed DDPG (TD3)** are two key RL algorithms designed for such environments.

While **DDPG** provides a strong baseline, it often suffers from **instability and overestimation bias**. **TD3** improves upon this by introducing:
✔ **Twin critics** (reducing overestimation bias)
✔ **Delayed policy updates** (improving stability)
✔ **Target smoothing** (enhancing exploration)

In this project, we implement **DDPG** and **TD3** in the **LunarLander-v2** environment, where an agent controls a spacecraft aiming for a **soft, fuel-efficient landing**.

* The **state space** includes **position, velocity, angle, and leg contact**.
* The **action space** involves **thruster control** for maneuvering.
* The **reward system** encourages smooth landings while penalizing crashes and excessive tilt.

By systematically testing different hyperparameters, we analyze how **policy delay, target noise, and critic structure** impact **stability, exploration, and learning efficiency**—ultimately demonstrating why **TD3 outperforms DDPG** in this scenario.

---

## 📄 Report

For a detailed explanation of the methodology, results, and conclusions, refer to the full report:

📥 [**Read the full report**](./RL_LunarLander_Report_AdrianSKazi.pdf)

---

## 🏃 Running the LunarLander Simulation

Before running the simulation, install the required dependencies:

```bash
pip install -r requirements.txt
```

then execute:

```bash
python LunarLander.py
```

---

## 🧠 Running (Modular Version)

Now the project is modular. Use `main.py` as entry point.

### ▶️ Train agent

```bash
cd Donor && python main.py --mode train
```

### 🎥 Test + generate video

```bash
cd Donor && python main.py --mode test
```

---

## 📂 Outputs

After training:

```bash
models_saved/actor.pth
mlruns/
```

After testing:

```bash
videos/rl-video-episode-0.mp4
```

---

## ⚙️ Notes

* `--mode train` → trains TD3 agent and logs metrics to MLflow
* `--mode test` → loads trained model and generates rollout video
* All configs are in `config/config.py`
