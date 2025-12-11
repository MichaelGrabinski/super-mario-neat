import os
import csv
import statistics as stats
import neat
import gym, ppaquette_gym_super_mario
import pickle
import multiprocessing as mp
import queue
import visualize
import neat.reporting
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

gym.logger.set_level(40)


class Train:
    def __init__(self, generations, parallel=2, level="1-1", preset=None):
        self.actions = [
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
        ]
        self.generations = generations
        self.lock = mp.Lock()
        self.sequential = False
        if os.name == 'nt':
            if parallel > 1:
                print("Windows detected; forcing parallel=1 to keep FCEUX stable.")
            self.par = 1
            self.sequential = True
        else:
            self.par = parallel
        self.level = level
        self.metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        self.generation_index = 0
        self.generation_offset = 0
        self.preset = preset
        self.best_fitness_so_far = None
        self.stats_reporter = neat.StatisticsReporter()
        self.tb_writer = None
        if SummaryWriter is not None:
            tb_dir = os.path.join(self.metrics_dir, "tensorboard", "neat")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

    def _get_actions(self, a):
        return self.actions[a.index(max(a))]

    def _evaluate_genome(self, genome_id, genome, config):
        env = gym.make('ppaquette/SuperMarioBros-' + self.level + '-Tiles-v0')
        try:
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            i = 0
            old = 40
            early_stop = False
            reward_sum = 0.0
            while not done:
                state = state.flatten()
                output = net.activate(state)
                output = self._get_actions(output)
                s, reward, done, info = env.step(output)
                info = info or {}
                state = s
                i += 1
                reward_sum += reward
                if i % 50 == 0:
                    distance = info.get('distance', 0)
                    if old == distance:
                        early_stop = True
                        break
                    else:
                        old = distance

            distance = info.get('distance', 0)
            fitness = -1 if distance <= 40 else distance
            genome.fitness = fitness
            if fitness >= 3252:
                pickle.dump(genome, open("finisher.pkl", "wb"))
                print("Finisher found; genome saved to finisher.pkl")
            metrics = {
                "genome_id": genome_id,
                "fitness": fitness,
                "distance": distance,
                "steps": i,
                "early_stop": early_stop,
                "reward_sum": reward_sum,
                "level": self.level,
            }
            return metrics
        except KeyboardInterrupt:
            raise
        finally:
            env.close()

    def _fitness_func(self, genome_id, genome, config, o):
        try:
            metrics = self._evaluate_genome(genome_id, genome, config)
        except Exception as e:
            # Surface the error and still return a metrics row so CSVs are not missing generations.
            print(f"Error evaluating genome {genome_id}: {e}")
            metrics = {
                "genome_id": genome_id,
                "fitness": -1,
                "distance": 0,
                "steps": 0,
                "early_stop": True,
                "reward_sum": 0.0,
                "level": self.level,
                "error": str(e),
            }
        o.put(metrics)

    def _eval_genomes(self, genomes, config):
        generation = self.generation_offset + self.generation_index
        self.generation_index += 1
        genomes = list(genomes)

        generation_logs = []
        print(f"Evaluating generation {generation} with {len(genomes)} genomes...")
        if self.sequential:
            for genome_id, genome in genomes:
                metrics = self._evaluate_genome(genome_id, genome, config)
                metrics["generation"] = generation
                generation_logs.append(metrics)
        else:
            for i in range(0, len(genomes), self.par):
                output = mp.Queue()

                chunk = genomes[i:i + self.par]
                chunk_map = {genome_id: genome for genome_id, genome in chunk}
                processes = [mp.Process(target=self._fitness_func, args=(genome_id, genome, config, output)) for
                             genome_id, genome in chunk]

                [p.start() for p in processes]
                [p.join() for p in processes]

                results = []
                for _ in processes:
                    try:
                        results.append(output.get(timeout=5))
                    except queue.Empty:
                        print("Warning: missing result from a worker process; skipping.")

                for result in results:
                    genome_id = result["genome_id"]
                    genome_obj = chunk_map.get(genome_id)
                    if genome_obj is not None:
                        genome_obj.fitness = result["fitness"]
                    result["generation"] = generation
                    generation_logs.append(result)

        self._write_generation_metrics(generation, generation_logs, genomes)
        self._prune_checkpoints(limit=5)

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        self.config = config
        self._apply_preset(config)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.Checkpointer(5))
        p.add_reporter(self.stats_reporter)
        p.add_reporter(self._artifact_reporter())
        print("Starting fresh population...")
        self.generation_offset = getattr(p, "generation", 0)
        winner = p.run(self._eval_genomes, n)
        win = p.best_genome
        pickle.dump(winner, open('winner.pkl', 'wb'))
        pickle.dump(win, open('real_winner.pkl', 'wb'))

        visualize.draw_net(config, winner, True)
        visualize.plot_stats(self.stats_reporter, ylog=False, view=False,
                             filename=os.path.join(self.metrics_dir, "fitness.svg"))
        visualize.plot_species(self.stats_reporter, view=False,
                               filename=os.path.join(self.metrics_dir, "species.svg"))

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._ensure_metrics_dir()
        self._run(config_path, self.generations)

    def _ensure_metrics_dir(self):
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _write_generation_metrics(self, generation, logs, genome_records):
        if not logs:
            print(f"No logs recorded for generation {generation}, skipping metrics write.")
            return
        path = os.path.join(self.metrics_dir, f"gen_{generation:04d}.csv")
        fieldnames = ["generation", "genome_id", "fitness", "distance", "steps", "early_stop", "reward_sum", "level", "error"]
        for entry in logs:
            entry.setdefault("error", "")
        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
        print(f"Wrote generation metrics to {path}")

        # Append to rolling all.csv
        all_path = os.path.join(self.metrics_dir, "all.csv")
        write_header = not os.path.exists(all_path)
        with open(all_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(logs)

        fitnesses = [entry["fitness"] for entry in logs]
        distances = [entry["distance"] for entry in logs]
        early_stops = sum(1 for entry in logs if entry["early_stop"])
        reward_sums = [entry["reward_sum"] for entry in logs]
        summary = (
            f"Gen {generation}: "
            f"best={max(fitnesses):.1f}, "
            f"mean={stats.mean(fitnesses):.1f}, "
            f"median={stats.median(fitnesses):.1f}, "
            f"std={stats.pstdev(fitnesses) if len(fitnesses) > 1 else 0:.1f}, "
            f"avg_distance={stats.mean(distances):.1f}, "
            f"avg_reward={stats.mean(reward_sums):.2f}, "
            f"early_stops={early_stops}/{len(logs)}"
        )
        print(summary)
        summary_row = {
            "generation": generation,
            "best_fitness": max(fitnesses),
            "mean_fitness": stats.mean(fitnesses),
            "median_fitness": stats.median(fitnesses),
            "std_fitness": stats.pstdev(fitnesses) if len(fitnesses) > 1 else 0,
            "avg_distance": stats.mean(distances),
            "avg_reward": stats.mean(reward_sums),
            "early_stops": early_stops,
            "population": len(genome_records),
        }
        if self.tb_writer:
            self.tb_writer.add_scalar("fitness/best", summary_row["best_fitness"], generation)
            self.tb_writer.add_scalar("fitness/mean", summary_row["mean_fitness"], generation)
            self.tb_writer.add_scalar("fitness/median", summary_row["median_fitness"], generation)
            self.tb_writer.add_scalar("fitness/std", summary_row["std_fitness"], generation)
            self.tb_writer.add_scalar("distance/avg", summary_row["avg_distance"], generation)
            self.tb_writer.add_scalar("reward/avg", summary_row["avg_reward"], generation)
            self.tb_writer.add_scalar("early_stops/count", early_stops, generation)
            self.tb_writer.flush()
        summary_path = os.path.join(self.metrics_dir, "summary.csv")
        summary_fields = list(summary_row.keys())
        write_summary_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
            if write_summary_header:
                writer.writeheader()
            writer.writerow(summary_row)

        # Save best genome of this generation and best overall.
        genome_map = {gid: g for gid, g in genome_records}
        best_entry = max(logs, key=lambda e: e["fitness"])
        best_genome = genome_map.get(best_entry["genome_id"])
        if best_genome is not None:
            fname = f"best_gen_{generation:04d}_fit{int(best_entry['fitness'])}.pkl"
            pickle.dump(best_genome, open(os.path.join(self.metrics_dir, fname), "wb"))
            if self.best_fitness_so_far is None or best_entry["fitness"] > self.best_fitness_so_far:
                self.best_fitness_so_far = best_entry["fitness"]
                pickle.dump(best_genome, open(os.path.join(self.metrics_dir, "best_overall.pkl"), "wb"))
            try:
                net_path = os.path.join(self.metrics_dir, f"net_gen_{generation:04d}")
                visualize.draw_net(self.config, best_genome, view=False, filename=net_path, prune_unused=True)
                print(f"Saved topology snapshot to {net_path}.svg")
            except Exception as e:
                print(f"Could not render topology for generation {generation}: {e}")

    def _prune_checkpoints(self, limit=5):
        # Keep only the newest N neat-checkpoint-* files.
        files = []
        for name in os.listdir(os.getcwd()):
            if name.startswith("neat-checkpoint-"):
                path = os.path.join(os.getcwd(), name)
                files.append((os.path.getmtime(path), path))
        files.sort(reverse=True)
        for _, path in files[limit:]:
            try:
                os.remove(path)
            except OSError:
                pass

    def _apply_preset(self, config):
        # Avoid NEAT complaining when many species spawn in generation 0.
        config.reproduction_config.min_species_size = 1
        # Fewer species explosions; tune if you want more/less speciation.
        if hasattr(config.species_set_config, "compatibility_threshold") and config.species_set_config.compatibility_threshold < 5:
            config.species_set_config.compatibility_threshold = 8.0
            print("Raised compatibility_threshold to 8.0 to reduce species explosion.")
        if self.preset is None:
            return
        if self.preset == "debug":
            config.pop_size = 20
            config.species_set_config.max_stagnation = 3
        elif self.preset == "fast":
            config.pop_size = 50
            config.species_set_config.max_stagnation = 8
        elif self.preset == "full":
            # use defaults
            pass

    def _artifact_reporter(self):
        trainer = self

        class ArtifactReporter(neat.reporting.BaseReporter):
            def post_evaluate(self, config, population, species, best_genome):
                try:
                    visualize.plot_stats(trainer.stats_reporter, ylog=False, view=False,
                                         filename=os.path.join(trainer.metrics_dir, "fitness.svg"))
                    visualize.plot_species(trainer.stats_reporter, view=False,
                                           filename=os.path.join(trainer.metrics_dir, "species.svg"))
                except Exception as e:
                    print(f"Could not write plots this generation: {e}")

        return ArtifactReporter()


if __name__ == "__main__":
    t = Train(1000)
    t.main()
