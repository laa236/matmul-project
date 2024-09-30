#!/share/apps/python/anaconda/bin/python

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plot(runs):
    "Plot results of timing trials"
    for arg in runs:
        df = pd.read_csv("timing-{0}.csv".format(arg))
        plt.plot(df['size'], df['mflop'] / 1e3, label=arg)
    plt.xlabel('Dimension')
    plt.ylabel('Gflop/s')
    df = pd.read_csv("timing-mine.csv")
    print("min: ", min(df["mflop"]))
    print("max: ", max(df["mflop"]))
    print("avg: ", df["mflop"].mean())

def show(runs):
    "Show plot of timing runs (for interactive use)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def main(runs):
    "Show plot of timing runs (non-interactive)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Determine the filename for saving
    base_filename = 'timing'
    file_extension = '.pdf'
    file_counter = 0
    new_filename = f"{base_filename}{file_extension}"

    # Check for existing files and generate a new filename if needed
    while os.path.exists(new_filename):
        file_counter += 1
        new_filename = f"{base_filename}{file_counter}{file_extension}"

    # Save the figure
    plt.savefig(new_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print(f"Plot saved as: {new_filename}")

if __name__ == "__main__":
    main(sys.argv[1:])