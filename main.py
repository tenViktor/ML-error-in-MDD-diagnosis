import os
import mne
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
Author: Viktor Morovič
2nd year biomedical engineering and bioinformatics student

VUTmail: 257026@vutbr.cz
MUNImail: 550786@mail.muni.cz

This is a project for my data analysis and visualization class @FIT VUT Brno.

NOTE: THIS SCRIPT REQUIRES THE DATA FOLDER TO BE PRESENT IN THIS DIRECTORY
NOTE: THIS SCRIPT REQUIRES WELL ANOTATED EEG FILES 
- FILENAME SHOULD INCLUDE WHETHER THE EYES WERE OPEN/CLOSED/P300 TASK
- THE ORIGINAL DATA DOES NOT HAVE ANNOTATIONS

Training and validation data have been manually split before being loaded in this script

Github repo: https://github.com/tenViktor/ML-error-in-MDD-diagnosis
Original dataset: https://figshare.com/articles/dataset/EEG_Data_New/4244171/2

"""

# Enabling interactive mode to display shap results correctly
plt.ion

def load_data(filename: str) -> pd.DataFrame:
    """
    Loads data from a directory and preprocesses it.
    
    Data gets read, cleaned and filtered using a low-pass filter to remove any noise
    
    
    :param str filename: name of the directory containing data
    :return pd.Dataframe: dataframe containing clean data
    """
    
    # defining path
    # Change path in case of different directory placement
    path = os.path.join(
        os.getcwd(),
        'ML-error-in-MDD-diagnosis',
        filename
        )

    # empty list for creating dataframe 
    df_list = list()
    
    # Initializing features
    state = None
    epoch = None
    good_channels = [
        'EEG F3-LE', 'EEG F4-LE', 'EEG Fz-LE',
        'EEG Cz-LE',             
        'EEG P3-LE', 'EEG P4-LE', 'EEG Pz-LE', 
        'EEG T3-LE', 'EEG T4-LE'         
    ]  

    for file in os.listdir(path):
        # Labeling Healthy and MDD samples
        if ('H S' in file):
            # state = 0 -> healthy
            state = 0
        else:
            # state = 1 -> MDD
            state = 1
            
        if ('EO' in file):
            # epoch = 0 -> eyes open during data aquisition
            epoch = 0
        elif ('EC' in file):
            # epoch = 1 -> eyes closed during data aquisition
            epoch = 1
        else:
            # epoch = 2 -> P300 data
            epoch = 2
            
        full_path = os.path.join(
            path, file
            )
        
        # Loading the data
        raw = mne.io.read_raw_edf(
            full_path
            )
        
        # Basic data pre-processing
        raw.pick(
            good_channels
            )
        raw.crop(tmin=1,tmax=4).load_data()
        # low-pass filter for removing noise
        filtered_sig = raw.filter(
            l_freq=None, h_freq=40, fir_design='firwin'
            )
        
        
        for channel_name in raw.info['ch_names']:
            channel_data = raw.get_data(
                picks=channel_name
                )
            df_list.append({
                'state': state, 'type': epoch, 
                'channel': channel_data,'data': filtered_sig}
                           )
    
    return pd.DataFrame(df_list)

if __name__ == '__main__':
    df = load_data('.training_data')
    df.style
    print('success')