#%%%
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
#%%%
# Enabling interactive mode to display shap results correctly
plt.ion

def _find_best_leaf(max_leaf_nodes: list, train_X:np.ndarray, val_X:np.ndarray, train_y:np.ndarray, val_y:np.ndarray, show_figure:bool=False):
    """
    Finds the best number of leaf nodes for Decision Trees
    
    :param list max_leaf_nodes: list of different numbers of leaves
    :param list train_X...val_y: split data
    :param bool show_figure: graph of leaf nodes and their accuracy scores
    :return int best_leaf: optimal number of leaf nodes 
    """
    
    best_acs = 0
    best_leaf = 0
    leaf_acs_list = list()
    
    for leaf_size in max_leaf_nodes:
        model = DecisionTreeClassifier(max_leaf_nodes=leaf_size, random_state=0)
        model.fit(train_X, train_y)
        pred_y = model.predict(val_X)
        acs = accuracy_score(val_y, pred_y)
        leaf_acs_list.append(acs)
        if best_acs != None:
            if acs > best_acs:
                best_acs = acs
                best_leaf = leaf_size
            else:
                continue
        else:
            best_acs = acs
            best_leaf = leaf_size
    
    
    if show_figure:
        graph = pd.DataFrame({'x': max_leaf_nodes, 'y': leaf_acs_list})
        graph['optimal'] = graph['x'] == best_leaf
        
        plt.figure(figsize=(12,8))
        sns.histplot(data=graph, hue='optimal', palette={True: 'red', False:'blue'})
        
        plt.title('Comparison of leaf nodes and their accuracy score')
        plt.xlabel('Number of leaf nodes')
        plt.ylabel('Accuracy score')
        plt.legend(labels=['non-optimal','optimal'])
    
    return best_leaf
    

def load_data(filename: str, epoch_length: float) -> pd.DataFrame:
    """
    Loads data from a directory and preprocesses it.
    
    Data gets read, cleaned and filtered using a low-pass filter to remove any noise
    
    
    :param str filename: name of the directory containing data
    :param float epoch_duration: duration of a single epoch 
    - used for signal splicing and calculations
    - smaller epoch_duration requires larger datasets in order to prevent overfitting
    :return pd.Dataframe: dataframe containing clean data
    """
    
    # defining path
    # Change path in case of different directory placement
    path = os.path.join(
        os.getcwd(),
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

    channel_dict = {channel: idx for idx, channel in enumerate(good_channels)}  
    print(channel_dict)
    
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
            epoch_type = 0
        elif ('EC' in file):
            # epoch = 1 -> eyes closed during data aquisition
            epoch_type = 1
        else:
            # epoch = 2 -> P300 data
            epoch_type = 2
            
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
        
        #cropping out the beginning in case artifacts and/or filter initialization
        raw.crop(tmin=1,tmax=4).load_data()
        # low-pass filter for removing noise
        raw.filter(
            l_freq=None, h_freq=40, 
            fir_design='firwin', verbose= None
            )
        
        sfreq = raw.info['sfreq']
        samples_per_epoch = int(sfreq*epoch_length)
        print(samples_per_epoch)
        
        # for each EEG channel
        for channel_name in raw.info['ch_names']:
            channel_data = raw.get_data(
                picks=channel_name
                )
            
            # calculating number of epochs based on individual epoch length
            epoch_num = len(channel_data) // samples_per_epoch
            epochs = np.array_split(
                channel_data[:epoch_num * samples_per_epoch],
                epoch_num
                )
            print(epoch_num)
            print(epochs)
            
            # defining each row in a DataFrame
            # feature definition
            for epoch in epochs:
                df_list.append({
                    'state': state, 
                    'type': epoch_type, 
                    'channel': channel_dict[channel_name],
                    'mean': np.mean(epoch),
                    'std': np.std(epoch),
                    'min': np.min(epoch),
                    'max': np.max(epoch),
                    'ptp': np.ptp(epoch),
                    'kurtosis': float(pd.Series(epoch).kurtosis())
                    }
                )
    
    return pd.DataFrame(df_list)

def split_data(data: pd.DataFrame) -> tuple:
    """
    Spits loaded data into training and validating data
    
    :param pd.DataFrame data: original DataFrame containing the data
    :return tuple: return training and validation data
    """
    
    # separating our features and function
    eeg_data = data.dropna(axis=0)
    eeg_features = ['type', 'channel', 'data']
    X = eeg_data[eeg_features]
    y = eeg_data.state

    # splitting data and ensuring consistend split
    return train_test_split(X,y, random_state=1)
    
    
#%%%
if __name__ == '__main__':
    #%%%
    df = load_data(filename='.training_data', epoch_length=0.5)
    #%%%
    df.style
    print('success')
    df.shape
    df
    
    #%%%
    train_X, val_X, train_y, val_y = split_data(df)
    display(train_X, val_X, train_y, val_y)
    #%%%
    _find_best_leaf([5,50,500,5000], train_X=train_X, val_X=val_X,train_y=train_y,val_y=val_y, show_figure=True)
# %%
