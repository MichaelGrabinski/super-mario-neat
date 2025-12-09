import argparse
import train as t
import run as r
import cont_train as ct

parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('mode', metavar='mode', type=str,
                    help="Specify 'train' or 'run' to run or train the model. To continue training, specify 'cont_train")
parser.add_argument('--gen', metavar='generations', type=int, help='Number of Generations to run for', nargs='?')
parser.add_argument('--file', metavar='file_name', type=str, help='File name to continue training or to run the winner',
                    nargs='?')
parser.add_argument('--config', metavar='config', type=str, help='Configuration File', default='config', nargs='?')
parser.add_argument('--parallel', metavar='parallel', type=int,
                    help='Number of genomes to run at once', nargs='?', default=2)
parser.add_argument('--level', metavar='level', type=str, help='Which level to run, Eg. 1-1', default='1-1', nargs='?')
parser.add_argument('--preset', metavar='preset', type=str, choices=['debug', 'fast', 'full'],
                    help='Training preset to override config values (debug/fast/full)', nargs='?')
parser.add_argument('--render', action='store_true', help='Render environment when running')
parser.add_argument('--max-episodes', metavar='max_episodes', type=int, default=50,
                    help='Max episodes to attempt when running')
parser.add_argument('--max-steps', metavar='max_steps', type=int, default=5000,
                    help='Max steps per episode when running')
parser.add_argument('--progress', action='store_true', help='Show per-episode progress when running')

args = parser.parse_args()

if (args.mode.upper() == "TRAIN" or args.mode.upper() == "CONT_TRAIN") and args.gen is None:
    parser.error("Please specify number of generations!")

if args.mode.upper() == "CONT_TRAIN" and args.file is None:
    parser.error("Please specify checkpoint file ("
                 "./Files/neat-checkpoint-2492 can be used to start from generation 2492)!")

if args.mode.upper() == "TRAIN":
    t = t.Train(args.gen, args.parallel, args.level, preset=args.preset)
    t.main(config_file=args.config)
elif args.mode.upper() == "CONT_TRAIN":
    c = ct.Train(args.gen, args.file, args.parallel, args.level, preset=args.preset)
    c.main(config_file=args.config)

elif args.mode.upper() == "RUN":
    args.file = "finisher.pkl" if args.file is None else args.file
    r.main(args.config, args.file, args.level, max_episodes=args.max_episodes,
           max_steps=args.max_steps, render=args.render, progress=args.progress)

else:
    print("Please enter 'train', 'run', or 'cont_train'")
