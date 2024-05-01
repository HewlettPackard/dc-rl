import os
import numpy as np
import pandas as pd

import os

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]


def obtain_paths(location):
    """Obtain the correct name for the data files

    Args:
        location (string): Location identifier

    Raises:
        ValueError: If location identifier is not defined

    Returns:
        List[string]: Naming for the data files
    """
    if 'az' in location.lower():
        return ['AZ', 'USA_AZ_Phoenix-Sky.Harbor.epw']
    elif 'ca' in location.lower():
        return ['CA', 'USA_CA_San.Jose-Mineta.epw']
    elif 'ga' in location.lower():
        return ['GA', 'USA_GA_Atlanta-Hartsfield-Jackson.epw']
    elif 'il' in location.lower():
        return ['IL', 'USA_IL_Chicago.OHare.epw']
    elif 'ny' in location.lower():
        return ['NY', 'USA_NY_New.York-LaGuardia.epw']
    elif 'tx' in location.lower():
        return ['TX', 'USA_TX_Dallas-Fort.Worth.epw']
    elif 'va' in location.lower():
        return ['VA', 'USA_VA_Leesburg.Exec.epw']
    elif "wa" in location.lower():
        return ['WA', 'USA_WA_Seattle-Tacoma.epw']
    else:
        raise ValueError("Location not found")

def get_energy_variables(state):
    """Obtain energy variables from the energy observation

    Args:
        state (List[float]): agent_dc observation

    Returns:
        List[float]: Subset of the agent_dc observation
    """
    energy_vars = np.hstack((state[4:7],(state[7]+state[8])/2))
    return energy_vars


# Function to get the initial index of the day of a given month from a time-stamped dataset
def get_init_day(start_month=0):
    """Obtain the initial day of the year to start the episode on

    Args:
        start_month (int, optional): Starting month. Defaults to 0.

    Returns:
        int: Day of the year corresponding to the first day of the month
    """
    assert 0 <= start_month <= 11, "start_month should be between 0 and 11 (inclusive, 0-based, 0=January, 11=December)."

    # Read the CSV file and parse dates from the 'timestamp' column
    df = pd.read_csv(PATH+'/data/CarbonIntensity/NY_NG_&_avgCI.csv', parse_dates=['timestamp'], usecols=['timestamp'])
    
    # Extract the month from each timestamp and add it as a new column to the DataFrame
    df['month'] = pd.DatetimeIndex(df['timestamp']).month
    
    # Find the first day of the specified start month
    init_day = df[df['month'] == start_month+1].index[0]
    
    # Return the day number (0-based)
    return int(init_day/24)
