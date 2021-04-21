import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def readFile(file):

    with open(file) as f:
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
file2 = "../Breast_data2.txt"
file3 = "../Crawl_data1.txt"

# read = readFile(file)
# read = readFile(file2)
read = readFile(file3)

df = pd.DataFrame(read)
df = df.drop('tfps', axis=1)

# df = df.drop(df.index[328:628]) # Breast data1
# df = df.drop(df.index[421:449]) # Breast data2
df = df.drop(df.index[359:445]) # Crawl data1

x = np.array(df)

y = np.zeros([len(df),1])

# y[127:186] = 1 # Breast data1
# y[175:243] = 1 # Breast data2
y[159:229] = 1 # Crawl data1

y_train = np.zeros([len(df),1])

for i in range(len(df)):
    if x[i,1] > 1:
        y_train[i] = 1

print(np.mean(y == y_train))

# --- Breast_data1 -----
# 0-127 breast
# 127-186 drowning
# 186-327 breast
# 327-628 standing

# --- Breast_data2 -----
# 0-175 breast
# 175-243 drowning
# 243-420 breast
# 420-448 standing

# --- Crawl_data1 -----
# 0-159 Crawl
# 159-229 drowning
# 229-358 breast
# 358-444 standing

# read = readFile(file3)
#
# df = pd.DataFrame(read)
# df = df.drop('tfps', axis=1)
#
# df.plot();
# plt.xlabel('time every 200ms');
# plt.legend(loc='best')
# plt.show()