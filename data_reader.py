import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

def readFile(infile):
    regex = re.compile('(?:\s*([+-]?\d*.\d*))')
    end = len(lines)
    
    with open(infile) as f:
        lines = f.readlines()
        lines = lines[:-2]

        x   = []
        y   = []
        z  = []
        tfps = []

    for line in lines[25:(end-3)]:
            linedata = regex.findall(line)
            x.append(float(linedata[0]))
            y.append(float(linedata[1]))
            z.append(float(linedata[2]))
            tfps.append(float(linedata[3]))


    data = {'x': np.array(x), 'y': np.array(y), 'z': np.array(z), 'tfps': np.array(tfps)}

    return data


bData1 = '/Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/Breast_data1'
bData2 = '/Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/Breast__1.txt'
cData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/Crawl__3.txt'

read = readFile(bData2)

df = pd.DataFrame(read)

df = df.drop('tfps', axis=1)


plt.figure(figsize=(15,8));
df.plot();
plt.xlabel('time every 200ms');
plt.legend(loc='best')
