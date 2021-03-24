import pandas as pd # version 1.0.5
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

def file_ext_to_list(empty_list, ext):
    for root, dirs_list, files_list in os.walk(dir):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == ext:
                empty_list.append(file_name)


def plot_dataframe(csvfilename, dataframe):
    # print(dataframe_motion.head())
    # plot on same axes
    plt.figure()
    ax = plt.gca()
    ax.plot(dataframe.index, dataframe.x, label='x-axis', color='red')
    ax.plot(dataframe.index, dataframe.y, label='y-axis', color='blue')
    ax.plot(dataframe.index, dataframe.z, label='z-axis', color='green')
    plt.title(csvfilename[:-4].replace('_', " ").title())
    plt.xlabel('time (sec)')
    plt.ylabel('position relative to origin')
    plt.legend()
    plt.show()


# Scale data between 0 to 1:
def scaleData(data):
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)

# get log file in list to iterate over later
# dir is directory where script is run from; also the directory which contains the log files
dir = '/Users/vedangjoshi/PycharmProjects/MDM3_Phase_C/Accelerometer_Data'
csv_extension = '.csv'

csv_file_name_list = []
file_ext_to_list(csv_file_name_list, csv_extension)

dataframe_list = []
for csvfile in csv_file_name_list:
    dataframe_motion = pd.read_csv(dir+'/'+csvfile, sep=';')
    dataframe_list.append(dataframe_motion)


for i,j in zip(csv_file_name_list, dataframe_list):
    plot_dataframe(csvfilename=i, dataframe=j)

scaled_dataframe_list = []
for i in dataframe_list:
    data_scaled = scaleData(i[['x','y','z']])
    x_array = []
    y_array = []
    z_array = []
    for i in range(len(data_scaled)):
        x_array.append(data_scaled[i][0])
        y_array.append(data_scaled[i][1])
        z_array.append(data_scaled[i][2])

    scaled_dataframe = pd.DataFrame(
        {'x': x_array,
         'y': y_array,
         'z': z_array
         })

    scaled_dataframe_list.append(scaled_dataframe)

for i, j in zip(csv_file_name_list, scaled_dataframe_list):
    plot_dataframe(csvfilename='scaled_'+i, dataframe=j)