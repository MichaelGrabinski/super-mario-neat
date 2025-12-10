import neat
import pickle
import gym, ppaquette_gym_super_mario
import visualize
import gzip
import neat.genome
import os
import types

ACTIONS = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
]
# FILENAME = "./Files/gen_2284"
CONFIG = 'config'


def main(config_file, file, level="1-1", max_episodes=50, max_steps=5000, render=False, progress=False):
    # with gzip.open(FILENAME) as f:
    #   config = pickle.load(f)[1]
    # print(str(config.genome_type.size))
    base_dir = os.path.dirname(__file__)
    config_path = config_file
    if not os.path.isabs(config_path):
        candidate = os.path.join(base_dir, config_path)
        config_path = candidate if os.path.isfile(candidate) else config_path

    file_path = file
    if not os.path.isabs(file_path):
        candidate = os.path.join(base_dir, file_path)
        file_path = candidate if os.path.isfile(candidate) else file_path

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
    info = {'distance': 0}
    try:
        for episode in range(max_episodes):
            if info.get('distance', 0) == 3252:
                break
            state = env.reset()
            done = False
            i = 0
            old = 40
            reward_sum = 0.0
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
                if i % 50 == 0:
                    distance = info.get('distance', 0)
                    if old == distance:
                        break

                    else:
                        old = distance
            if progress:
                print(f"Episode {episode+1}: distance={info.get('distance', 0)}, "
                      f"steps={i}, reward_sum={reward_sum:.2f}")
        else:
            print(f"Stopped after {max_episodes} episodes without reaching goal distance.")
        if info.get('distance', 0) == 3252:
            print("Goal distance reached.")
        env.close()
    except KeyboardInterrupt:
        env.close()
        exit()


if __name__ == "__main__":
    main(CONFIG, "finisher.pkl")
