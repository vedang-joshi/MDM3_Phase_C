''' This file will format our data so it can be used in the wavelet transform and CNN '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_reader import readFile


bData1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/Breast_data1.txt'
bData2 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/breast__1.txt'
cData1 = '/Users/jakebeard/Documents/GitHub/MDM3_Phase_C/crawl__3.txt'

bData1, tbData1 = readFile(bData1)
bData2, tbData2 = readFile(bData2)
cData1, tcData1 = readFile(cData1)

print(tbData1)


data = list(bData1.items())
an_array = np.array(data)
print(an_array)
df = pd.DataFrame(bData1)
remainder = len(df) % 60
df.drop(df.tail(remainder).index, inplace = True)

print(len(df), remainder)
df = df.drop('tfps', axis=1)


plt.figure(figsize=(15,8));
df.plot();
plt.xlabel('time every 200ms');
plt.legend(loc='best')
print(df)
