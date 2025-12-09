import argparse
import pickle

import gym, ppaquette_gym_super_mario
import neat

try:
    import imageio
except ImportError:
    imageio = None

ACTIONS = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
]


def replay(config_file, file, level="1-1", max_steps=5000, render=False, record=None):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = gym.make('ppaquette/SuperMarioBros-' + level + '-Tiles-v0')
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    frames = []
    info = {'distance': 0}
    reward_sum = 0.0
    try:
        state = env.reset()
        done = False
        i = 0
        old = 40
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
            if record:
                frame = env.render(mode='rgb_array')
                frames.append(frame)
            if i % 50 == 0:
                if old == info.get('distance', 0):
                    break
                else:
                    old = info.get('distance', 0)
        print(f"Replay finished: distance={info.get('distance', 0)}, steps={i}, reward_sum={reward_sum:.2f}")
        if record:
            if imageio is None:
                print("imageio not installed; cannot write recording.")
            elif not frames:
                print("No frames captured; nothing to write.")
            else:
                imageio.mimsave(record, frames, fps=30)
                print(f"Saved recording to {record}")
        env.close()
    except KeyboardInterrupt:
        env.close()
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a trained genome and optionally record a GIF.")
    parser.add_argument('--config', type=str, default='config', help='Config file path')
    parser.add_argument('--file', type=str, default='finisher.pkl', help='Genome pickle to replay')
    parser.add_argument('--level', type=str, default='1-1', help='Level to play (e.g., 1-1)')
    parser.add_argument('--max-steps', type=int, default=5000, help='Max steps to run')
    parser.add_argument('--render', action='store_true', help='Render during replay')
    parser.add_argument('--record', type=str, help='Path to save GIF (requires imageio)')
    args = parser.parse_args()
    replay(args.config, args.file, level=args.level, max_steps=args.max_steps, render=args.render, record=args.record)
