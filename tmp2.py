import gymnasium as gym
from elegantrl.train.config import Config
from elegantrl.agents.AgentPPO import AgentDiscretePPO
from elegantrl.train.run import train_agent
from multiprocessing import freeze_support

env = gym.make

env_args = {
    "id": "CartPole-v1",
    "env_name": "CartPole-v1",
    "num_envs": 1,
    "max_step": 1000,
    "state_dim": 4,
    "action_dim": 2,
    "if_discrete": True,
    "reward_scale": 2**-1,
}
args = Config(AgentDiscretePPO, env_class=env, env_args=env_args)
args.gpu_id = 1
args.max_step = 1000
args.reward_scale = 2**-1  # RewardRange: -1800 < -200 < -50 < 0
args.gamma = 0.99
args.target_step = args.max_step
args.eval_times = 2**3
args.num_workers=4 #rollout, improve gpu utilization
args.num_threads=4
# args.learner_gpu_ids=[0,1]
args.net_dims = [64, 64]  # the middle layer dimension of MLP (MultiLayer Perceptron)
args.learning_rate = 3e-4*2  # the learning rate for network updating
args.use_tensorboard=True
args.scheduler_name = 'WarmupCosineLR'
args.break_step = 1e5
args.scheduler_args = {
    'total_steps': args.break_step,
    'warmup_steps_pct': 0.1, # e.g., 10% of total scheduler activations for warmup
    'warmup_start_factor': 0.01,
    'cosine_end_factor': 0.05,
}


if __name__ == '__main__':
    freeze_support()
    train_agent(args)
    