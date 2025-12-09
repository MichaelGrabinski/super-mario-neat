import neat
import pickle
import gym, ppaquette_gym_super_mario
import visualize
import gzip
import neat.genome

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
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = gym.make('ppaquette/SuperMarioBros-'+level+'-Tiles-v0')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = {'distance': 0}
    try:
        for episode in range(max_episodes):
            if info['distance'] == 3252:
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
                state = s
                i += 1
                reward_sum += reward
                if render:
                    env.render()
                if i % 50 == 0:
                    if old == info['distance']:
                        break

                    else:
                        old = info['distance']
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
