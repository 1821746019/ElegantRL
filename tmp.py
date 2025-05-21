import gymnasium as gym
from elegantrl.train.config import Config
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.run import train_agent
from multiprocessing import freeze_support

env = gym.make

env_args = {
    "id": "Pendulum-v1",
    "env_name": "Pendulum-v1",
    "num_envs": 1,
    "max_step": 1000,
    "state_dim": 3,
    "action_dim": 1,
    "if_discrete": False,
    "reward_scale": 2**-1,
    "gpu_id": 0, # if you have GPU
}
args = Config(AgentPPO, env_class=env, env_args=env_args)
args.max_step = 1000
args.reward_scale = 2**-1  # RewardRange: -1800 < -200 < -50 < 0
args.gamma = 0.99
args.target_step = args.max_step
args.eval_times = 2**3
args.num_workers=16 #rollout, improve gpu utilization
args.num_threads=4
# args.learner_gpu_ids=[0,1]
args.net_dims = [64, 64]  # the middle layer dimension of MLP (MultiLayer Perceptron)
args.learning_rate = 3e-3*10  # the learning rate for network updating
args.use_tensorboard=True
args.scheduler_name = 'WarmupCosineLR'
# Calculate total number of optimizer updates (scheduler steps)

# if args.num_workers > 0 and args.horizon_len > 0:
#     # This logic assumes the multiprocessing path via train_agent_multiprocessing
#     # where steps towards break_step are accumulated by (horizon_len * num_workers) per learner update cycle.
#     effective_env_steps_per_scheduler_activation = args.horizon_len * args.num_workers
#     if effective_env_steps_per_scheduler_activation == 0: # Should not happen with valid horizon_len and num_workers
#         total_scheduler_activations = int(args.break_step) # Fallback, though problematic
#     else:
#         total_scheduler_activations = int(args.break_step / effective_env_steps_per_scheduler_activation)
# else:
#     # Fallback for single process or if num_workers/horizon_len is not set as expected (e.g. 0)
#     # For single process, effective_env_steps_per_scheduler_activation would be args.horizon_len
#     if args.horizon_len > 0:
#         total_scheduler_activations = int(args.break_step / args.horizon_len)
#     else:
#         total_scheduler_activations = int(args.break_step) # Fallback

args.scheduler_args = {
    'max_steps': 1e6,
    'warmup_steps': 1e6*0.1, # e.g., 10% of total scheduler activations for warmup
    'warmup_start_factor': 0.01,
    'lr_min': 0.001 * args.learning_rate,
}


if __name__ == '__main__':
    freeze_support()
    train_agent(args)
    