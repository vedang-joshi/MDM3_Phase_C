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

split_by_secs = 60
remainder = len(tbData1) % split_by_secs
tbData1 = tbData1[:-remainder]
split_data = np.array([tbData1[x:x+split_by_secs] for x in range(0, len(tbData1), split_by_secs)])
split_data.transpose()
print(split_data)
