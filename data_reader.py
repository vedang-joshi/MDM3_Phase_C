import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

def readFile(infile):
    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(infile) as f:
        lines = f.readlines()
        lines = lines[:-2]

        x   = []
        y   = []
        z  = []
        tfps = []

    for line in lines[25:]:
            linedata = regex.findall(line)
            x.append(float(linedata[0]))
            y.append(float(linedata[1]))
            z.append(float(linedata[2]))
            tfps.append(float(linedata[3]))


    data = {'x': np.array(x), 'y': np.array(y), 'z': np.array(z), 'tfps': np.array(tfps)}

    return data


bData1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Breast_data1.txt'
bData2 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/breast__1.txt'
cData1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/crawl__3.txt'

read = readFile(bData2)

df = pd.DataFrame(read)
print(df)

df = df.drop('tfps', axis=1)


plt.figure(figsize=(15,8));
df.plot();
plt.xlabel('time every 200ms');
plt.legend(loc='best')
plt.show()
