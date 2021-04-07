def readFile(infile):
    import re
    import numpy as np

    regex = re.compile('(?:\s*([+-]?\d*.\d*))')

    with open(infile) as f:
        lines = f.readlines()
        lines = lines[:-2]

        transformed_data = []
        x   = []
        y   = []
        z  = []
        tfps = []

    for line in lines[25:]:
            linedata = regex.findall(line)
            transformed_data.append(np.array([float(linedata[0]),float(linedata[1]),float(linedata[2]),float(linedata[3])]))
            x.append(float(linedata[0]))
            y.append(float(linedata[1]))
            z.append(float(linedata[2]))
            tfps.append(float(linedata[3]))


    data = {'x': np.array(x),'y': np.array(y), 'z': np.array(z), 'tfps': np.array(tfps)}
    np.asarray(transformed_data)

    return data, transformed_data
