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
        
# finds the optimal number of leaves for Decision Tree Classifier 
def _find_best_leaf(
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    prune: bool = False,
    show_figure: bool = False
) -> tuple:
    """
    Finds the best number of leaf nodes for Decision Trees with optional pruning
    
    
    :param np.ndarray train_X: Training feature
    :param np.ndarray train_y: Training label
    :param np.ndarray val_X: validation feature
    :param np.ndarray val_y: validation label
    :param bool prune: Whether to apply cost complexity pruning
    :param bool show_figure: Whether to display performance graphs
    
    :return tuple: (best_leaf_nodes, best_ccp_alpha) if pruning enabled, else best_leaf_nodes
    """
    max_leaf_nodes = np.arange(start=50, stop=5500, step=500)
    
    # Base hyperparameters
    if prune:
        min_leaf = 4
        min_split = 5
    else:
        min_leaf = 1
        min_split = 2
    base_params = {
        'min_samples_leaf': min_leaf,
        'min_samples_split': min_split,
        'random_state': 0
    }
    
    # Defining score lists
    train_scores = list()
    val_scores = list()
    
    for leaf_size in max_leaf_nodes:
        model = DecisionTreeClassifier(
            max_leaf_nodes=leaf_size, **base_params
            )
        model.fit(train_X, train_y)
        
        train_scores.append(
            accuracy_score(train_y, model.predict(train_X)
                           ))
        val_scores.append(
            accuracy_score(val_y, model.predict(val_X))
            )
    
    # Find best leaf size based on validation performance and overfitting penalty
    delta = np.array(train_scores) - np.array(val_scores)
    
    # penalization for large delta
    overall_score = np.array(val_scores) - 0.3 * delta
    best_leaf = max_leaf_nodes[np.argmax(overall_score)]
    
    if not prune:
        if show_figure:
            _plot_learning_curves(
                max_leaf_nodes, train_scores, val_scores, 
                title='Comparison of leaf nodes and their accuracy score - no pruning'
                )
        return best_leaf
    
    # Pruning the tree
    model = DecisionTreeClassifier(
        max_leaf_nodes=best_leaf, **base_params
        )
    path = model.cost_complexity_pruning_path(
        train_X, train_y
        )
    
    # Removing n-th alpha - prunes the whole tree
    ccp_alphas = path.ccp_alphas[:-1]  
    
    # Find best alpha through cross-validation
    pruned_train_scores = []
    pruned_val_scores = []
    
    for ccp_alpha in ccp_alphas:
        pruned_model = DecisionTreeClassifier(
            max_leaf_nodes=best_leaf,
            ccp_alpha=ccp_alpha,
            **base_params
        )
        pruned_model.fit(train_X, train_y)
        
        pruned_train_scores.append(
            accuracy_score(train_y, pruned_model.predict(train_X))
            )
        pruned_val_scores.append(
            accuracy_score(val_y, pruned_model.predict(val_X))
            )
    
    best_alpha_idx = np.argmax(pruned_val_scores)
    best_ccp_alpha = ccp_alphas[best_alpha_idx]
    
    if show_figure:
        # Plot original learning curves
        _plot_learning_curves(
            max_leaf_nodes, train_scores, val_scores,
            title='Leaf nodes comparison - before pruning'
            )
        
        # Plot learning curves after pruning
        _plot_learning_curves(
            ccp_alphas,
            pruned_train_scores,
            pruned_val_scores,
            x_label='CCP Alpha',
            title=f'Pruning the tree using optimal leaf size ({best_leaf})'
            )
        
        # Plot SHAP values for final model
        _plot_shap_values(
            train_X,
            train_y,
            best_leaf,
            best_ccp_alpha,
            base_params
            )
    
    return best_leaf, best_ccp_alpha

# Helper func
def _plot_learning_curves(
    x_values, train_scores, val_scores, 
    x_label='Number of leaf nodes',
    title='Learning Curves'
    ):
    """Helper function to plot learning curves."""
    plt.figure(figsize=(12, 8))
    graph_df = pd.DataFrame({
        x_label: np.concatenate([x_values, x_values]),
        'Accuracy': train_scores + val_scores,
        'Type': ['Training'] * len(x_values) + ['Validation'] * len(x_values)
    })
    
    sns.lineplot(
        data=graph_df,
        x=x_label,
        y='Accuracy',
        hue='Type',
        style='Type',
        palette='tab10',
        markers=True,
        marker='o'
    )
    plt.title(title)
    plt.ylabel('Accuracy score')
    plt.legend()

# Helper func for shap - shap doesn't work
def _plot_shap_values(X, y, leaf_nodes, ccp_alpha, base_params):
    """Helper function to plot SHAP values."""
    try:
        plt.figure(figsize=(12, 8))
        model = DecisionTreeClassifier(
            max_leaf_nodes=leaf_nodes,
            ccp_alpha=ccp_alpha,
            **base_params
        )
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        shap.plots.bar(shap_values)
    except IndexError:
        pass    
    
# Loading the data from dataset       
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
        raw.crop(tmin=0.5, tmax=5).load_data()
        # low-pass filter for removing noise
        raw.filter(
            l_freq=None, h_freq=40, 
            fir_design='firwin', verbose= None
            )
        
        sfreq = raw.info['sfreq']
        samples_per_epoch = int(sfreq*epoch_length)
        
        # for each EEG channel
        for channel_name in raw.info['ch_names']:
            channel_data = raw.get_data(
                picks=channel_name
                )[0]
            
            
            
            # calculating number of epochs based on individual epoch length
            epoch_num = len(channel_data) // samples_per_epoch
            epochs = np.array_split(
                channel_data[:epoch_num * samples_per_epoch],
                epoch_num
                )

            
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

# splitting data into training data
def split_data(data: pd.DataFrame) -> tuple:
    """
    Spits loaded data into training and validating data
    
    :param pd.DataFrame data: original DataFrame containing the data
    :return tuple: return training and validation data
    """
    
    # separating our features and function
    eeg_data = data.dropna(axis=0)
    eeg_features = [
                    'type',
                    'channel',
                    'mean',
                    'std',
                    'min',
                    'max',
                    'ptp',
                    'kurtosis'
                    ]
    X = eeg_data[eeg_features]
    y = eeg_data.state

    # splitting data and ensuring consistend split
    return train_test_split(
        X,y,
        test_size=0.2,
        train_size=0.8,
        random_state=1
        )
 
# TODO:  
# random forest    
# def random_forest(best_leaf):
#
#SGDclassifier
# def sgd_class()
#
# Try out regressors and fuzzy logic 

# FIXME:
# look at shap IndexError issue
    

if __name__ == '__main__':
    df = load_data(filename='.training_data', epoch_length=0.125)
    train_X, val_X, train_y, val_y = split_data(df)
    
    # Without pruning
    best_leaf = _find_best_leaf(
        train_X, val_X,
        train_y, val_y,
        show_figure=True
        )
    
    # With pruning
    best_leaf, best_ccp_alpha = _find_best_leaf(
        train_X, val_X,
        train_y, val_y, 
        prune=True, show_figure=True)
    print(best_leaf, best_ccp_alpha)
    # I got too lost in trying to fix overfitting, that unfortunately I ran out of time
