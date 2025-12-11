import neat
import pickle
import gym, ppaquette_gym_super_mario
import visualize
import gzip
import neat.genome
import os
import csv
import time
import datetime

try:
    import imageio
except ImportError:
    imageio = None

ACTIONS = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
]
# FILENAME = "./Files/gen_2284"
CONFIG = 'config'


def main(config_file, file, level="1-1", max_episodes=50, max_steps=5000, render=False, progress=False,
         fps=60.0, record=None, log_file=None, topology=None):
    base_dir = os.path.dirname(__file__)
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = config_file
    if not os.path.isabs(config_path):
        candidate = os.path.join(base_dir, config_path)
        config_path = candidate if os.path.isfile(candidate) else config_path

    file_path = file
    if not os.path.isabs(file_path):
        candidate = os.path.join(base_dir, file_path)
        file_path = candidate if os.path.isfile(candidate) else file_path

    log_path = log_file
    if log_path is None:
        log_path = os.path.join(metrics_dir, f"run_{timestamp}.csv")

    topology_path = topology
    if topology_path is None:
        topology_path = os.path.join(metrics_dir, f"run_net_{timestamp}")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Support gzip-compressed files (e.g., neat checkpoints)
    with open(file_path, 'rb') as fh:
        start = fh.read(2)
    if start == b'\x1f\x8b':
        loader = gzip.open
    else:
        loader = open

    loaded = pickle.load(loader(file_path, 'rb'))
    # If this is a NEAT checkpoint tuple, grab the best genome in the population.
    genome = loaded
    if isinstance(loaded, tuple) and len(loaded) >= 3:
        # loaded: (generation, config, population, species, rndstate)
        try:
            population = loaded[2]
            genome = max(population.values(), key=lambda g: g.fitness or -1)
            print("Loaded genome from checkpoint population.")
        except Exception as e:
            raise RuntimeError(f"Unsupported checkpoint format; provide a pickled genome instead. Details: {e}")
    env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    if topology_path:
        try:
            visualize.draw_net(config, genome, view=False, filename=topology_path, prune_unused=True)
            print(f"Saved topology to {topology_path}.svg")
        except Exception as e:
            print(f"Could not save topology: {e}")
    info = {'distance': 0}
    episode_logs = []
    frame_delay = 0.0 if fps is None or fps <= 0 else 1.0 / fps
    try:
        for episode in range(max_episodes):
            if info.get('distance', 0) == 3252:
                break
            state = env.reset()
            done = False
            i = 0
            old = 40
            reward_sum = 0.0
            frames = []
            while not done and i < max_steps:
                state = state.reshape(208)
                output = net.activate(state)
                ind = output.index(max(output))
                s, reward, done, info = env.step(ACTIONS[ind])
                info = info or {}
                state = s
                i += 1
                reward_sum += reward
                if render:
                    env.render()
                if record:
                    try:
                        frame = env.render(mode='rgb_array')
                        frames.append(frame)
                    except Exception as e:
                        print(f"Could not capture frame: {e}")
                        record = None
                if frame_delay > 0 and render:
                    time.sleep(frame_delay)
                if i % 50 == 0:
                    distance = info.get('distance', 0)
                    if old == distance:
                        break

                    else:
                        old = distance
            episode_logs.append({
                "episode": episode + 1,
                "distance": info.get('distance', 0),
                "steps": i,
                "reward_sum": reward_sum,
                "level": level,
                "file": file_path,
            })
            if record:
                if imageio is None:
                    print("imageio not installed; cannot write recording.")
                    record = None
                elif not frames:
                    print("No frames captured; nothing to write.")
                    record = None
                else:
                    imageio.mimsave(record, frames, fps=max(1, int(fps or 30)))
                    print(f"Saved recording to {record}")
                    record = None  # only record first episode unless overridden
            if progress:
                print(f"Episode {episode+1}: distance={info.get('distance', 0)}, "
                      f"steps={i}, reward_sum={reward_sum:.2f}")
        else:
            print(f"Stopped after {max_episodes} episodes without reaching goal distance.")
        if info.get('distance', 0) == 3252:
            print("Goal distance reached.")
        if episode_logs:
            fieldnames = ["episode", "distance", "steps", "reward_sum", "level", "file"]
            with open(log_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(episode_logs)
            print(f"Wrote run metrics to {log_path}")
        env.close()
    except KeyboardInterrupt:
        env.close()
        exit()


if __name__ == "__main__":
    main(CONFIG, "finisher.pkl")
