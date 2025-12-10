# Super Mario NEAT
This program uses the NEAT algorithm to evolve a 
Neural Network to play the original Super Mario Bros.
<br>


## Requirements
You can install the requirements by running <br >
```bash 
sudo apt install fceux
python3 -m pip install -r requirements.txt
```

Or if on windows, run
```bash
python3 -m pip install -r requirements.txt
```
* Make sure you have FCEUX downloaded and added to PATH

## Training
The finisher.pkl file contains the best genome on generation 2284.
In ./Files, you can find the backup for generation 2284, and the backup for generation 2492,
which is where I stopped training. <br />
<br>
You can continue training by running <br>
```bash
python3 main.py cont_train --gen <num_generations> --file <file>
```
<br>

On Windows, run commands from the repo root and point to the src config implicitly:
```bash
python src/main.py train --gen 10 --level 1-1
```
Per-generation metrics and topology svgs are written to `src/metrics/` (requires Graphviz installed system-wide). You'll see lines like `Wrote generation metrics to src/metrics/gen_0000.csv` and `Saved topology snapshot to src/metrics/net_gen_0000.svg`.
On Windows the training pool runs single-process for stability; you can still set `--gen` and other flags normally.

## Running
To run the finisher.pkl file, run
<br>
```bash
python3 main.py run
```

or run <br>
```bash
python3 run.py
```
If you want to run a different file, run<br>
```bash
python3 main.py run --file <file_name>
```
From the repo root on Windows, include the src prefix:
```bash
python src/main.py run --render
```
Run looks for config and pickle files relative to `src/` when not given absolute paths.
To run a neat checkpoint directly (compressed or not), point at it with `--file`; the best genome in the checkpoint population is used.
<br>

## Config
For debugging values, you can change any of the values in the config file. Note that you have to train from the 1st generation for some to take effect.
<br>
To use a different config file when training, specify `--config <config file>` when running `main.py`.
<br>
## Multiprocessing
This program uses the build in python module multiprocessing, which is used for parallel computing. You can adjust the amount of genomes
to run at once by specifying `--parallel <num_of_genomes>` when running `main.py`.
<br>
## Levels
The default level is World 1, Level 1. This can be changed by specifying `--level <level>` when running `main.py`. For example, <br>
`python3 main.py train --gen 100 --level "1-1"` will use 1-1.
<br>
## Result
The `finisher.pkl` file is trained to complete 1-1. It can complete it around 50% of the time. The `run.py` file keeps running the
simulation until it completes the level. `Ctrl + C` will stop it.
<br>
<br>
<img src="https://github.com/vivek3141/super-mario-neat/raw/master/img/world1-1.gif">
<br>

## Additional Information
The [Wiki](https://github.com/vivek3141/super-mario-neat/wiki) contains more information regarding the specifics of implementing certain parts.
