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

df = df.drop('tfps', axis=1)
df = df.drop(df.index[328:628])

# print(df)

x = np.array(df)

y = np.zeros([len(df),1])
y[127:186] = 1

y_train = np.zeros([len(df),1])

for i in range(len(df)):
    if x[i,1] > 1:
        y_train[i] = 1

print(np.mean(y == y_train))


# 0-127 breast
# 127-186 drowning
# 186-327 breast
# 327-628 standing

# df = pd.read_csv("Breast_data1.txt")
# df = df[23:-1]
# print(df[0])
# df.plot()

# df.plot();
# plt.xlabel('time every 200ms');
# plt.legend(loc='best')
# plt.show()