import os
import mne
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

def _find_best_leaf(max_leaf_nodes: list, train_X:list, val_X:list, train_y:list, val_y:list):
    """
    Finds the best number of leaf nodes for Decision Trees
    
    :param list max_leaf_nodes: list of different numbers of leaves
    :param list train_X...val_y: split data
    :return int best_leaf: optimal number of leaf nodes 
    """
    
    best_acs = 0
    best_leaf = 0
    
    for leaf_size in max_leaf_nodes:
        model = DecisionTreeClassifier(max_leaf_nodes=leaf_size, random_state=0)
        model.fit(train_X, train_y)
        pred_y = model.predict(val_X)
        acs = accuracy_score(val_y, pred_y)
        if best_acs != None:
            if acs > best_acs:
                best_acs = acs
                best_leaf = leaf_size
            else:
                continue
        else:
            best_acs = acs
            best_leaf = leaf_size
    
    return best_leaf
    
    # getting accuracy score
    acs = accuracy_score(val_y, pred_y)

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

def split_data(data: pd.DataFrame) -> tuple:
    """
    Spits loaded data into training and validating data
    
    :param pd.DataFrame data: original DataFrame containing the data
    :return tuple: return training and validation data
    """
    
    # separatin our features and function
    eeg_data = data.dropna(axis=0)
    eeg_features = ['type', 'channel', 'data']
    X = eeg_features[eeg_features]
    y = eeg_data.state

    # splitting data and ensuring consistend split
    train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)
    
    return train_X, val_X, train_y, val_y

if __name__ == '__main__':
    df = load_data('.training_data')
    df.style
    print('success')
    train_X, val_X, train_y, val_y = split_data(df)