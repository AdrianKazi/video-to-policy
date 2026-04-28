# # Adrian's AE
Z_DIM = 64

# Environment
ENV_NAME = "LunarLanderContinuous-v3"
STATE_DIM = 8
ACTION_DIM = 2
MAX_ACTION = 1.0

# GAIfO training
TOTAL_ITERATIONS = 300        # Note: 160 seems to be enough to get a good rough idea
ROLLOUT_STEPS = 2048 
MAX_EP_STEPS = 900            # Episode length

# Disc.
DISC_HIDDEN_DIM = 256
DISC_LR = 1e-4
DISC_EPOCHS = 5              # Had decent results with 3 and 5
DISC_BATCH_SIZE = 512

# PPO Policy Parms.
# Ref: https://github.com/warrenzha/ppo-gae-pytorch/blob/97f7bb338227321b218bea5aa6a16bcc23c8618e/env/config.py#L36POLICY_HIDDEN_DIM = 256
POLICY_HIDDEN_DIM = 256
POLICY_LR = 1e-4
VALUE_LR = 1e-4
PPO_EPOCHS = 10             
PPO_BATCH_SIZE = 64
PPO_CLIP = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.003
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5

# Log stuff
EVAL_INTERVAL = 10
EVAL_EPISODES = 5
LOG_INTERVAL = 5

