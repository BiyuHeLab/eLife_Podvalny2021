
import numpy as np
from scipy.io import loadmat
import pandas as pd
import HLTP_pupil

all_df = pd.read_pickle(HLTP_pupil.result_dir +'/all_subj_bhv_df.pkl')

def print_block_duration():
    n_trials_in_block = 36
    mean_block_duration = []
    for subject in HLTP_pupil.subjects:
        data_dir = HLTP_pupil.MEG_raw_dir + '/' + subject
        bhv = loadmat(data_dir +'/Behavior/datafile.mat')['data']
        trial_onset = bhv['timing'][0][0][0]['chOnset'][0][0]
        trial_complete = bhv['timing'][0][0][0]['trialCompleteTime'][0][0]
        n_trials = trial_onset.shape[0]
        block_duration = trial_complete[np.arange(n_trials_in_block - 1, n_trials, n_trials_in_block)
                         ] - trial_onset[np.arange(0, n_trials, n_trials_in_block)]
        mean_block_duration.append(block_duration.mean())
    mean_block_duration = np.array(mean_block_duration)/60 #minutes
    print("Mean block duration ", np.mean(mean_block_duration), '+-', np.std(mean_block_duration)/np.sqrt(len(HLTP_pupil.subjects)))
