import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def readFile(file):

    with open("Breast_data1.txt") as f:
        regex = re.compile('(?:\s*([+-]?\d*.\d*))')

        lines = f.readlines()
        lines = lines[:-2]
        x = []
        y = []
        z = []
        tfps = []

    for line in lines[25:]:
        linedata = regex.findall(line)
        x.append(float(linedata[0]))
        y.append(float(linedata[1]))
        z.append(float(linedata[2]))
        tfps.append(float(linedata[3]))

    data = {'x': np.array(x), 'y': np.array(y), 'z': np.array(z), 'tfps': np.array(tfps)}

    return data

file = "Breast_data1.txt"

read = readFile(file)

df = pd.DataFrame(read)
print(df)

df = df.drop('tfps', axis=1)

# df = pd.read_csv("Breast_data1.txt")
# df = df[23:-1]
# print(df[0])
# df.plot()

