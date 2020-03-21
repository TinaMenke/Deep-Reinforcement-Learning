import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
from baselines.common import plot_util as pu

### Set your path to the folder containing the .csv files
PATH='your .csv file path' # Use your path

### Fetch all files in path
fileNames=os.listdir(PATH)

### Filter file name list for files ending with .csv
fileNames=[file for file in fileNames if '.csv' in file]
print(fileNames)

### Loop over all files:
for file in fileNames:
    df=pd.read_csv(PATH+file)
    x=df.Step
    y=df.Value
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('discounted reward in training 100000 timesteps')
    plt.plot(x,y,linewidth=1)

plt.show()
