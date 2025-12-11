import datetime
import os
from typing import Optional

import numpy as np
import gym, ppaquette_gym_super_mario

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    SB3_AVAILABLE = True
except Exception as e:  # pragma: no cover - only runs when SB3 missing
    SB3_AVAILABLE = False
    _IMPORT_ERROR = e

ACTIONS = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
]


def _require_sb3():
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 (and PyTorch) not installed. "
            "Install with `pip install \"stable-baselines3==1.8.0\" torch` inside your venv. "
            f"Original error: {_IMPORT_ERROR}"
        )


class FlattenObservation(gym.ObservationWrapper):
    """Flattens the Mario tile observation into a 1D float vector for MLP policies."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(208,), dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).flatten()


class DiscreteActions(gym.ActionWrapper):
    """Maps a discrete action index to the multi-binary controller vector used by the environment."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

    def action(self, action):
        return ACTIONS[int(action)]


def make_env(level="1-1", log_dir: Optional[str] = None, render: bool = False):
    def _init():
        env = gym.make(f'ppaquette/SuperMarioBros-{level}-Tiles-v0')
        env = FlattenObservation(env)
        env = DiscreteActions(env)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env

    return _init


def train_ppo(
    total_timesteps: int,
    level: str = "1-1",
    model_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    eval_freq: int = 10000,
):
    """Train a PPO agent on Mario and save the model plus monitor/eval logs."""
    _require_sb3()
    base_dir = os.path.dirname(__file__)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = log_dir or os.path.join(base_dir, "metrics", "ppo", timestamp)
    os.makedirs(metrics_dir, exist_ok=True)
    model_path = model_path or os.path.join(base_dir, "ppo_models", f"ppo_mario_{level}_{timestamp}.zip")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = DummyVecEnv([make_env(level, log_dir=metrics_dir)])
    eval_env = DummyVecEnv([make_env(level)])
    callback = EvalCallback(
        eval_env,
        best_model_save_path=metrics_dir,
        log_path=metrics_dir,
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=metrics_dir)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    model.save(model_path)
    print(f"PPO model saved to {model_path}")
    print(f"Training logs and evals saved to {metrics_dir}")
    return model_path


def run_ppo(
    model_path: str,
    level: str = "1-1",
    episodes: int = 5,
    max_steps: int = 5000,
    render: bool = False,
    log_file: Optional[str] = None,
):
    _require_sb3()
    if not os.path.isabs(model_path):
        candidate = os.path.join(os.path.dirname(__file__), model_path)
        model_path = candidate if os.path.isfile(candidate) else model_path
    model = PPO.load(model_path)
    env = make_env(level)()
    episode_logs = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                env.render()
        episode_logs.append(
            {
                "episode": ep + 1,
                "reward": total_reward,
                "steps": steps,
                "distance": info.get("distance", 0) if info else 0,
                "level": level,
                "model": model_path,
            }
        )
        print(f"PPO Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, distance={episode_logs[-1]['distance']}")
    env.close()
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(os.path.dirname(__file__), "metrics", "ppo", f"run_{timestamp}.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    import csv

    with open(log_file, "w", newline="") as csvfile:
        fieldnames = ["episode", "reward", "steps", "distance", "level", "model"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_logs)
    print(f"Wrote PPO run metrics to {log_file}")

