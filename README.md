# Evolutionary Parallel Tempering for Bayesian Neural Learning.

This work is based on G3PCX algorithm created by Prof K. Deb from IIT Kanpur.
Python implementation of the algorithm is available [here](https://github.com/rohitash-chandra/G3PCX-evoalg-py
"G3PCX").

## Running Evolutionary Parallel Tempering
```
    cd evolutionary-PT-MCMC
    python evo_pt_mcmc.py --problem <problem-name>
```

The script supports following additional command line arguments:
```
usage: evo_pt_mcmc.py [-h] [--problem PROBLEM] [--num-chains NUM_CHAINS]
                      [--population-size POPULATION_SIZE]
                      [--num-samples NUM_SAMPLES]
                      [--swap-interval SWAP_INTERVAL] [--burn-in BURN_IN]
                      [--max-temp MAX_TEMP] [--topology TOPOLOGY]
                      [--run-id RUN_ID] [--root ROOT]
                      [--train-data TRAIN_DATA] [--test-data TEST_DATA]
                      [--config-file CONFIG_FILE]

Run Evolutionary Parallel Tempering

optional arguments:
  -h, --help            show this help message and exit
  --problem PROBLEM     Problem to be used for Evolutionary PT: "synthetic",
                        "iris", "ions", "cancer", "bank", "penDigit", "chess", "TicTacToe"
  --num-chains NUM_CHAINS
                        Number of Chains for Parallel Tempering
  --population-size POPULATION_SIZE
                        Population size for G3PCX Evolutionary algorithm.
  --num-samples NUM_SAMPLES
                        Total number of samples (all chains combined).
  --swap-interval SWAP_INTERVAL
                        Number of samples between each swap.
  --burn-in BURN_IN     Ratio of samples to be discarded as burn-in samples.
                        Value 0.1 means 10 percent samples are discarded
  --max-temp MAX_TEMP   Temperature to be assigned to the chain with maximum
                        temperature.
  --topology TOPOLOGY   String representation of network topology. Eg:-
                        "input,hidden,output"
  --run-id RUN_ID       Unique Id to identify run.
  --root ROOT           path to root directory (evolutionary-pt).
  --train-data TRAIN_DATA
                        Path to the train data
  --test-data TEST_DATA
                        Path to the test data
  --config-file CONFIG_FILE
                        Path to data config yaml file
```

*Note: The default values of these values can be changed from config.py and data.yaml file*

## DataSets
Following Datasets are provided:
1. [Iris](https://archive.ics.uci.edu/ml/datasets/iris)
2. [Ionosphere](https://archive.ics.uci.edu/ml/datasets/ionosphere)
3. [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)
4. [Pen Digit](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
5. [Chess]()
6. [Bank]()
7. [Tic Tac Toe](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
