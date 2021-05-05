import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import math

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
file4 = "../breast_drowning1.txt"
file5 = "../breast_swimming1.txt"
file6 = "../breast__1.txt"
file7 = "../crawl__3.txt"
file8 = "../crawl_drowning1.txt"
file9 = "../crawl_swimming1.txt"

vedang_drown = "../Vedang_Accelerometer_Crawl/26-28Apr_accelerometer_heart_rate_data/crawl__25_drowning.txt"
vedang_swim = "../Vedang_Accelerometer_Crawl/26-28Apr_accelerometer_heart_rate_data/crawl__25_swimming.txt"
heart_file_loc = "../Vedang_Accelerometer_Crawl/26-28Apr_accelerometer_heart_rate_data/heart_rate_data.csv"


drown_heart = [98, 99, 100]
swim_heart = [70, 70, 70, 116, 114, 108, 101, 96, 94, 92, 91, 95]


# read = readFile(file)
# read = readFile(file2)
# read = readFile(file3)
read = readFile(vedang_drown)
read2 = readFile(vedang_swim)

df = pd.DataFrame(read)
df = df.drop('tfps', axis=1)
df2 = pd.DataFrame(read2)
df2 = df2.drop('tfps', axis=1)

# df = df.drop(df.index[328:628]) # Breast data1
# df = df.drop(df.index[421:449]) # Breast data2
# df = df.drop(df.index[359:445]) # Crawl data1

N = len(df)/len(drown_heart)
N2 = len(df2)/len(swim_heart)
print(N)
print(N2)

x = np.array(df)

x2 = np.array(df2)

y = np.zeros([len(df),1])
y2 = np.zeros([len(df2),1])

# y set to one when drowning
# y[127:186] = 1 # Breast data1
# y[175:243] = 1 # Breast data2
# y[159:229] = 1 # Crawl data1
y[:] = 1

y_train = np.zeros([len(df),1])
y2_train = np.zeros([len(df2),1])

print(math.floor(730/N))
for i in range(len(df)):
    if x[i,1] > 1 and drown_heart[math.floor(i/N)] > 90:
        y_train[i] = 1
for i in range(len(df2)):
    if x2[i,1] > 1 and swim_heart[math.floor(i/N2)] > 90:
        y2_train[i] = 1


mean1 = np.mean(y == y_train)
mean2 = np.mean(y2 == y2_train)
# print(np.mean(y == y_train))
print(np.mean([mean1, mean2]))

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

# ------------- Plotting Section ----------------------------
# read = readFile(vedang_swim)
#
# df = pd.DataFrame(read)
# df = df.drop('tfps', axis=1)
#
# # xx = [0, len(df)]
# # yy = [1, 1]
#
# df.plot()
# # plt.plot(xx,yy, linewidth=3, label="Threshold Line")
# plt.xlabel('Time (in 200ms steps)', fontsize=12)
# plt.ylabel('Acceleration (m/s^2)', fontsize=12)
# # plt.title("Plot of Raw Data with Threshold Line", fontsize=13)
# plt.legend(loc='best')
# plt.show()
# ------------- Plotting Section ----------------------------