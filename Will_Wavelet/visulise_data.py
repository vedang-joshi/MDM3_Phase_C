import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def readFile(infile):
    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(infile) as f:
        lines = f.readlines()

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

cData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/Crawl__3.txt'

read = readFile(cData1)

df = pd.DataFrame(read)
df.head()

total_ms = df.iloc[:,3].sum()
len_steps = len(df.iloc[:,3])
av_timestep = int(total_ms / len_steps)



# df = df.drop('tfps', axis=1)
#
# plt.figure(figsize=(15,8));
# df.plot();
# plt.xlabel('time every 200ms');
# plt.legend(loc='best')





# plt.style.use('seaborn')
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)

plt.style.use('fivethirtyeight')

def animation(i):
    cData1 = '//Users/will/Documents/UNI/MDM/MDM3/Drown/iDrown/Crawl__3.txt'

    data = readFile(cData1)
    df = pd.DataFrame(data)
    total_ms = df.iloc[:, 3].sum()
    len_steps = len(df.iloc[:, 3])
    av_timestep = int(total_ms / len_steps)

    x1 = df.iloc[5:, 0].tolist()
    y1 = df.iloc[5:, 1].tolist()
    z1 = df.iloc[5:, 2].tolist()
    tfps1 = df.iloc[5:, 3].tolist()

    x = x1[0:i+1]
    y = y1[0:i]
    z = z1[0:i]
    tfps = []
    for j in range(i+1):
        tfps.append(av_timestep*(j+1))
    print(x, tfps)

    plt.cla()
    plt.plot(tfps, x)
    # ax.clear()
    # ax.plot(tfps, x)
    # # ax.plot(tfps, y)
    # # ax.plot(tfps, z)


ani = FuncAnimation(plt.gcf(), animation, 1000)
plt.tight_layout()


# animation = FuncAnimation(fig, func=animation, interval=1000)
plt.show()

