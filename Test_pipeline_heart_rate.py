# code run on Mac, change dir name on line 17 to run on windows

import pandas as pd # version 1.0.5
import matplotlib.pyplot as plt
import os
from itertools import accumulate

def file_ext_to_list(empty_list, ext):
    for root, dirs_list, files_list in os.walk(dir):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == ext:
                empty_list.append(file_name)

# get log file in list to iterate over later
# dir is directory where script is run from; also the directory which contains the log files
dir = '/Users/vedangjoshi/PycharmProjects/MDM3_Phase_C/Heart_Rate_Data'
log_extension = '.log'
csv_extension = '.csv'

log_file_name_list = []
file_ext_to_list(log_file_name_list, log_extension)

# Convert log file to csv and save to list
for logfile in log_file_name_list:
    df = pd.read_fwf(dir+'/'+logfile)
    df.to_csv(logfile[:-4]+'.csv')

# get csv file in list to iterate over later
csv_file_name_list = []
file_ext_to_list(csv_file_name_list, csv_extension)


# Get continuous time series heart rate and plot
dataframe_list = []
for csvfile in csv_file_name_list:
    dataframe_log = pd.read_csv(dir+'/'+csvfile)
    # get two dataframes with offset and heart rate
    dataframe_offset = dataframe_log[dataframe_log['version : 3.11.3'].str.contains('offset')]
    dataframe_log = dataframe_log[dataframe_log['version : 3.11.3'].str.contains('hr_value')]

    # drop unnamed col
    dataframe_offset = dataframe_offset.drop(columns=['Unnamed: 0'])
    dataframe_log = dataframe_log.drop(columns=['Unnamed: 0'])
    #rename cols
    dataframe_offset = dataframe_offset.rename(columns={"version : 3.11.3": "offset"})
    dataframe_log = dataframe_log.rename(columns={"version : 3.11.3": "heart_rate"})

    # Data processing: remove rogue entries for heart rate
    dataframe_log = dataframe_log[~dataframe_log.heart_rate.str.contains("max")]
    dataframe_log = dataframe_log[~dataframe_log.heart_rate.str.contains("avg")]
    dataframe_offset = dataframe_offset[~dataframe_offset.offset.str.contains("_")]

    # reset indices and concat dataframes
    dataframe_log = dataframe_log.reset_index(drop=True)
    dataframe_offset = dataframe_offset.reset_index(drop=True)
    result = pd.concat([dataframe_log, dataframe_offset], axis=1)

    # extract heart rate and offset values from strings in cols in dataframe and drop NAN values
    result['heart_rate'] = result['heart_rate'].str.extract('(\d+)')
    result['offset'] = result['offset'].str.extract('(\d+)')
    result = result.apply(lambda x: pd.Series(x.dropna().values))

    # convert strings to ints and create a time axis from the offset values
    # in the form 0, 0+offset_1, 0+offset_1+offset_2 ...
    result['heart_rate'] = pd.to_numeric(result['heart_rate'])
    result['offset'] = pd.to_numeric(result['offset'])
    result['time'] = list(accumulate(result['offset']))

    dataframe_list.append(result)

# plot on same axes
ax = plt.gca()
for i in range(len(dataframe_list)):
    dataframe_list[i].plot(kind='line',x='time',y='heart_rate', ax=ax, alpha=0.6, label=csv_file_name_list[i][:-4])
    plt.ylabel('heart rate (bpm)')
    plt.xlabel('time (sec)')
plt.show()
