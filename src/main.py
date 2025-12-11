import argparse
import train as t
import run as r
import cont_train as ct
import ppo_agent as ppo

def main():
    parser = argparse.ArgumentParser(description='Run the program')
    parser.add_argument('mode', metavar='mode', type=str,
                        help="Specify 'train' or 'run' to run or train the model. To continue training, specify 'cont_train")
    parser.add_argument('--gen', metavar='generations', type=int, help='Number of Generations to run for', nargs='?')
    parser.add_argument('--file', metavar='file_name', type=str,
                        help='File name to continue training or to run the winner',
                        nargs='?')
    parser.add_argument('--config', metavar='config', type=str, help='Configuration File', default='config', nargs='?')
    parser.add_argument('--parallel', metavar='parallel', type=int,
                        help='Number of genomes to run at once', nargs='?', default=2)
    parser.add_argument('--level', metavar='level', type=str, help='Which level to run, Eg. 1-1', default='1-1',
                        nargs='?')
    parser.add_argument('--preset', metavar='preset', type=str, choices=['debug', 'fast', 'full'],
                        help='Training preset to override config values (debug/fast/full)', nargs='?')
    parser.add_argument('--render', action='store_true', help='Render environment when running')
    parser.add_argument('--max-episodes', metavar='max_episodes', type=int, default=50,
                        help='Max episodes to attempt when running')
    parser.add_argument('--max-steps', metavar='max_steps', type=int, default=5000,
                        help='Max steps per episode when running')
    parser.add_argument('--progress', action='store_true', help='Show per-episode progress when running')
    parser.add_argument('--fps', metavar='fps', type=float, default=60.0,
                        help='Target FPS when rendering runs; set to 0 for unlimited speed')
    parser.add_argument('--record', metavar='record', type=str, help='Path to save a GIF of the first episode')
    parser.add_argument('--log-file', metavar='log_file', type=str,
                        help='Path to write run metrics CSV (defaults to src/metrics/run_<timestamp>.csv)')
    parser.add_argument('--topology', metavar='topology', type=str,
                        help='Path (without extension) to save network topology svg during run')
    parser.add_argument('--timesteps', metavar='timesteps', type=int, default=50000,
                        help='PPO training timesteps')
    parser.add_argument('--ppo-model', metavar='ppo_model', type=str,
                        help='Path to save/load PPO model')
    parser.add_argument('--ppo-logdir', metavar='ppo_logdir', type=str,
                        help='Directory for PPO logs/metrics')
    parser.add_argument('--ppo-eval-freq', metavar='ppo_eval_freq', type=int, default=10000,
                        help='Eval frequency (timesteps) for PPO')
    parser.add_argument('--ppo-episodes', metavar='ppo_episodes', type=int, default=5,
                        help='Number of episodes when running PPO')

    args = parser.parse_args()

    if (args.mode.upper() == "TRAIN" or args.mode.upper() == "CONT_TRAIN") and args.gen is None:
        parser.error("Please specify number of generations!")

    if args.mode.upper() == "CONT_TRAIN" and args.file is None:
        parser.error("Please specify checkpoint file ("
                     "./Files/neat-checkpoint-2492 can be used to start from generation 2492)!")

    if args.mode.upper() == "TRAIN":
        trainer = t.Train(args.gen, args.parallel, args.level, preset=args.preset)
        trainer.main(config_file=args.config)
    elif args.mode.upper() == "CONT_TRAIN":
        c = ct.Train(args.gen, args.file, args.parallel, args.level, preset=args.preset)
        c.main(config_file=args.config)

    elif args.mode.upper() == "RUN":
        args.file = "finisher.pkl" if args.file is None else args.file
        r.main(args.config, args.file, args.level, max_episodes=args.max_episodes,
               max_steps=args.max_steps, render=args.render, progress=args.progress, fps=args.fps,
               record=args.record, log_file=args.log_file, topology=args.topology)
    elif args.mode.upper() == "PPO_TRAIN":
        ppo.train_ppo(
            total_timesteps=args.timesteps,
            level=args.level,
            model_path=args.ppo_model,
            log_dir=args.ppo_logdir,
            eval_freq=args.ppo_eval_freq,
        )
    elif args.mode.upper() == "PPO_RUN":
        model_path = args.ppo_model
        if model_path is None:
            parser.error("Please provide --ppo-model to load a PPO model.")
        ppo.run_ppo(
            model_path=model_path,
            level=args.level,
            episodes=args.ppo_episodes,
            max_steps=args.max_steps,
            render=args.render,
            log_file=args.log_file,
        )

    else:
        print("Please enter 'train', 'run', or 'cont_train'")


if __name__ == "__main__":
    main()
