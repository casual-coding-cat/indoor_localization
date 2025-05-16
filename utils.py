import pandas as pd
import numpy as np
import re

"""
this python file contains a copy of the functions created in the notebooks which are needed in other notebooks as well
"""

def read_data(path):
    # 1) Compile two regexes: one for the line header, one for each MAC=signal,channel,type
    line_re = re.compile(
        r't=(?P<time>\d+);'
        r'id=(?P<scanMac>[^;]+);'
        r'pos=(?P<posX>[^;]+);'
        r'degree=(?P<orientation>[^;]+);'
        r'(?P<responses>.+)'
    )
    resp_re = re.compile(
        r'(?P<mac>[^=;]+)='
        r'(?P<signal>-?\d+),'
        r'(?P<channel>\d+),'
        r'(?P<type>\d+)'
    )
    def process_line(line):
        """
        Parse one line of the offline trace into a list of
        [time, scanMac, posX, posY, posZ, orientation, mac, signal, channel, type].
        """

        m = line_re.match(line.strip())
        if not m:
            return []

        # extract header fields
        time        = m.group('time')
        scanMac     = m.group('scanMac')
        posX, posY, posZ = m.group('posX').split(',')
        orientation = m.group('orientation')

        # split off each "MAC=signal,channel,type" piece
        rows = []
        for resp in m.group('responses').split(';'):
            rm = resp_re.match(resp)
            if rm:
                rows.append([
                    time,
                    scanMac,
                    posX,
                    posY,
                    posZ,
                    orientation,
                    rm.group('mac'),
                    rm.group('signal'),
                    rm.group('channel'),
                    rm.group('type'),
                ])
        return rows

    # 2) Apply to every non-commented line in the file
    all_rows = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            all_rows.extend(process_line(line))

    # 3) Build the DataFrame
    columns = [
        "time", "scanMac",
        "posX", "posY", "posZ",
        "orientation",
        "mac", "signal",
        "channel", "type"
    ]
    df = pd.DataFrame(all_rows, columns=columns)

    # Now `df` has one signal per row, ready for cleaning, EDA, modeling…
    print(f"Parsed {len(df)} measurements.")
    return df


def round_orientation(angles):
    nearest_section = np.round(angles / 45)
    rounded = int(nearest_section* 45) % 360
    return rounded

def preprocessing(dataframe):
    # 1) Convert numeric variables from strings to appropriate numeric types
    numeric_cols = ['time', 'orientation', 'signal', 'channel', 'type']
    for col in numeric_cols:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

    # 2) Remove duplicate rows
    dataframe = dataframe.drop_duplicates()

    # 3) Keep only infrastructure APs (type==3) and drop the `type` column
    dataframe = dataframe[dataframe['type'] == 3].drop(columns=['type'])

    # 4) Convert `time` from milliseconds since epoch to datetime
    DF_TIME_UNIT = 'ms'  # adjust if needed
    dataframe.rename(columns={'time': 'rawTime'}, inplace=True) # rename to match reference
    dataframe['time'] = pd.to_datetime(dataframe['rawTime'], unit=DF_TIME_UNIT)

    # 5) remove MACs that don't occur often
    mac_counts = dataframe["mac"].value_counts()
    reference_aps = mac_counts.sort_values(ascending=False).head(7).index.tolist()
    print(f"reference APs: {reference_aps}")
    dataframe = dataframe[dataframe["mac"].isin(reference_aps)]

    # 6) transform coordinates
    dataframe['posXY'] = dataframe['posX'].astype(str) + '-' + dataframe['posY'].astype(str)
    dataframe = dataframe.drop(columns=['posZ'])
    dataframe["posX"] = pd.to_numeric(dataframe["posX"], errors='coerce')
    dataframe["posY"] = pd.to_numeric(dataframe["posY"], errors='coerce')

    # 7) round orientation to 45 degree sections
    dataframe['orientation_rounded'] = dataframe['orientation'].apply(round_orientation)

    return dataframe


def signal_summary(data):
    # Group the data by location (posXY), angle, and MAC address
    grouped = data.groupby(['posXY', 'orientation_rounded', 'mac'])

    data_summary = []

    for name, group in grouped:
        summary = group.iloc[0]  # Take the first row of each group
        summary['medSignal'] = np.median(group['signal'])
        summary['avgSignal'] = np.mean(group['signal'])
        summary['num'] = len(group['signal'])
        summary['sdSignal'] = np.std(group['signal'])
        summary['iqrSignal'] = np.percentile(group['signal'], 75) - np.percentile(group['signal'], 25)
        data_summary.append(summary)

    return pd.DataFrame(data_summary)

