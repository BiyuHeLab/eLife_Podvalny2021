#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:49:56 2019

@author: podvae01
"""

def meg_cross_corr(meg_data, pupil_data):
    ''' pearson corr at all lags 
    '''
    n_channels = meg_data.shape[0]
    n_times = meg_data.shape[1]
    cc = np.zeros( (n_channels, n_times * 2 - 1) )
    x2 = zscore(pupil_data)
    for chan in range(n_channels):
        x1 = zscore(meg_data[chan,:])
        cc[chan, :] = np.true_divide(
                np.correlate(x1, x2, "full"), 
                np.sqrt(np.dot(x2, x2)) * 
                np.sqrt(np.dot(x1, x1))) 
    return cc

# just all lags pearson
block_name = 'rest01'; cc = []
for subject in subjects:
    MEG_file = MEG_pro_dir + '/' + subject + '/'+ block_name + \
                '_ds_raw.fif'
    Pupil_file = MEG_pro_dir + '/' + subject + \
                        '/clean_interp_pupil_ds_' + block_name + '.pkl'
    pupil_data = HLTP_pupil.load(Pupil_file)
    
    raw = mne.io.read_raw_fif(MEG_file, preload = True)
    raw.pick_types(meg = True, ref_meg = False)
    meg_data = raw.get_data()
    
    cc.append(meg_cross_corr(meg_data, pupil_data))
cc_save = cc.copy() 
mean_lag_cc = np.arctanh(np.array(cc)).mean(axis = 0)
nsamp = mean_lag_cc.shape[1]
time = np.arange(-(nsamp - 1)/2, 1 + (nsamp - 1)/2) / HLTP_pupil.resamp_fs

for chan in range(5):
    plt.plot(mean_lag_cc[chan, :])    
ind = np.unravel_index(np.argmax(mean_lag_cc, axis=None), mean_lag_cc.shape)
best_chan = ind[0]; best_time = time[ind[1]]
mne.viz.plot_topomap(mean_lag_cc[:, ind[1]], pos, vmin = -0.2, vmax = 0.2)
plt.plot(time, mean_lag_cc[best_chan, :])   

ind = np.unravel_index(np.argmin(mean_lag_cc, axis=None), mean_lag_cc.shape)
best_chan = ind[0]; best_time = time[ind[1]]
mne.viz.plot_topomap(mean_lag_cc[:, ind[1]], pos, vmin = -0.2, vmax = 0.2)
plt.plot(time, mean_lag_cc[best_chan, :])   
mne.viz.plot_topomap(mean_lag_cc[:, 81109], pos, vmin = -0.2, vmax = 0.2)

# selected lags pearson
timelags = np.linspace(-1500, 1500, 31)   
lags = np.round(HLTP_pupil.resamp_fs * timelags / 1000).astype('int')
block_name = 'rest01';cc=[]
for subject in subjects:
    #for block_name in block_names:
    MEG_file = MEG_pro_dir + '/' + subject + '/'+ block_name + \
                '_ds_raw.fif'
    Pupil_file = MEG_pro_dir + '/' + subject + \
                        '/clean_interp_pupil_ds_' + block_name + '.pkl'
    pupil_data = HLTP_pupil.load(Pupil_file)
    
    raw = mne.io.read_raw_fif(MEG_file, preload = True)
    raw.pick_types(meg = True, ref_meg = False)
    meg_data = raw.get_data()
    
    cc.append(meg_cross_corr_sampled(meg_data, pupil_data, lags))






# plot example of pupil data processing
from scipy.signal import *
subject = 'AC'
plt.plot(pupil_data[150000:200000]);
plt.plot(pupil_data_clean[150000:200000]);
plt.ylim([-3.2, -2.8])

plt.plot(pupil_data[150100:160000]);
plt.plot(pupil_data_clean[150100:160000]);
plt.ylim([-3.2, -2.8])


# plot pupil and brain example
sub_pro_dir = MEG_pro_dir + '/' + subject
epochs = HLTP_pupil.get_raw_epochs(subject, epoch_name)
pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                 + '/' + subject + '/clean_pupil.pkl')
trial_number = 62
sample = epochs.events[trial_number, 0]   
one_trial_pupil = pupil_data[sample:(sample+2401)]
chan_number = 140
#plt.plot(zscore(epochs.get_data()[trial_number][chan_number]))
#plt.plot(zscore(one_trial_pupil))

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

x1 = zscore(epochs.get_data()[trial_number][chan_number])
x2 = zscore(one_trial_pupil)
plt.plot(butter_lowpass_filter(x1, 5, 1200, order=5))
plt.plot(butter_lowpass_filter(x2, 5, 1200, order=5))





