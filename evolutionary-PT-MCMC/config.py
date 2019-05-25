import argparse
import os, sys
from datetime import datetime
import multiprocessing as mp

# CREATE ARGUMENT PARSER
parser = argparse.ArgumentParser(description='Run Evolutionary Parallel Tempering')

# ADD ARGUMENTS TO THE PARSER
parser.add_argument('--problem', type=str, default='synthetic', help='Problem to be used for Evolutionary PT: \n"synthetic", "iris", "ions", "cancer", "bank", "PenDigit", "Chess"')
parser.add_argument('--num-chains', type=int, default=mp.cpu_count()-2, help='Number of Chains for Parallel Tempering')
parser.add_argument('--population-size', type=int, default=200, help='Population size for G3PCX Evolutionary algorithm.')
parser.add_argument('--num-samples', type=int, default=None, help='Total number of samples (all chains combined).')
parser.add_argument('--swap-interval', type=int, default=10, help='Number of samples between each swap.')
parser.add_argument('--burn-in', type=float, default=0.2, help=r'Ratio of samples to be discarded as burn-in samples. Value 0.1 means 10% samples are discared')
parser.add_argument('--max-temp', type=float, default=5, help='Temperature to be assigned to the chain with maximum temperature.')
parser.add_argument('--topology', type=str, default=None, help='String representation of network topology. Eg:- "input,hidden,output"')
parser.add_argument('--run-id', type=str, default="_".join(str(datetime.now()).split())
, help="Unique Id to identify run.")

# PARSE COMMANDLINE ARGUMENTS
opt = parser.parse_args()
print(opt)