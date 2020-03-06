#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:33:01 2019

@author: podvae01
"""
import HLTP_pupil
import mne
import pandas as pd  
import numpy as np                         
from HLTP_pupil import subjects, freq_bands, MEG_pro_dir
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs, read_beamformer
from mne.cov import compute_covariance
import scipy
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import signal
figures_dir = MEG_pro_dir  +'/_figures'
results_dir = MEG_pro_dir  +'/results'
bhv_dataframe = pd.read_pickle(MEG_pro_dir + 
                               '/results/all_bhv_pupil_df.pkl')
epoch_name = 'rest_pupil2s'
method = 'LCMV'
fs_ds = 2**8
# calculate data covariance subject  
for subject in subjects:
    print(['Make LCMV filter for subject '+ subject])

    sub_pro_dir = MEG_pro_dir + '/' + subject
    epochs = HLTP_pupil.get_raw_epochs(subject, epoch_name)
    epochs.resample(fs_ds)
    
    data_cov = mne.compute_covariance(epochs, tmin = 0., tmax = 2.,
                                  method = 'shrunk', rank = None)
    data_cov.save(sub_pro_dir + '/' + epoch_name + '-cov.fif')

    # create LCM  beamformer for each subjectt    
    #for subject in subjects:
    #sub_pro_dir = MEG_pro_dir + '/' + subject
    #data_cov = mne.read_cov(sub_pro_dir + '/' + epoch_name + '-cov.fif')
    noise_cov = mne.read_cov(sub_pro_dir + '/empty_room_for_rest1-cov.fif')
    fwd = mne.read_forward_solution(sub_pro_dir + '/HLTP_fwd.fif')
    info = epochs.info
    #info = mne.io.read_info(sub_pro_dir + '/' + epoch_name + '-epo.fif')
    filters = make_lcmv(info, fwd, data_cov, noise_cov = noise_cov,
                        reg = 0.05, pick_ori = 'max-power',
                    weight_norm = 'nai', rank = None)
    filters.save(sub_pro_dir + '/filters_'  + epoch_name + '-lcmv.h5', 
                 overwrite = True)

# Prepare single trials of MEG data in labels
subjects_dir = '/isilon/LFMI/VMdrive/Ella/mne_data/MNE-sample-data/subjects/'
labels_rh = mne.read_labels_from_annot(
        'fsaverage', 'HCPMMP1_combined', 'rh', subjects_dir=subjects_dir)
labels_lh = mne.read_labels_from_annot(
        'fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)
labels = labels_rh[1:] + labels_lh[1:]
n_labels = len(labels)
src_fname = subjects_dir + 'fsaverage/bem/fsaverage-ico-5-src.fif'
src = mne.read_source_spaces(src_fname)

band = 'beta'
for subject in subjects:
    print(['Source reconstruction in labels for subject '+ subject])

    sub_pro_dir = MEG_pro_dir + '/' + subject
    epochs = HLTP_pupil.get_raw_epochs(subject, epoch_name)
    
    epochs.filter(13, 30)
    epochs.apply_hilbert(envelope = True)

    epochs.resample(fs_ds)
    #epochs.save(sub_pro_dir + '/' + epoch_name + '_ds-epo.fif')
    filters = read_beamformer(sub_pro_dir + '/filters_' 
                              + epoch_name + '-lcmv.h5')  
    epochs.events[:, 2] = 0 # we don't care about pupil size across trials here

    stc = apply_lcmv_epochs(epochs, filters, max_ori_out='signed')
    morph = mne.compute_source_morph(stc[0], subject_from = subject,
                                 subject_to='fsaverage',
                                 subjects_dir=HLTP_pupil.MRI_dir)
    n_trials = len(epochs.events)
    fs = epochs.info['sfreq']
    n_times =  np.int(fs * 2)
    label_data = np.zeros((n_labels, n_trials, n_times))
    
    for trial_number in range( n_trials ):
        stc_fsaverage = morph.apply(stc[trial_number])
        labels_ts = mne.extract_label_time_course(stc_fsaverage, labels, src)
        label_data[:, trial_number, :] = labels_ts
        
    HLTP_pupil.save(label_data, HLTP_pupil.MEG_pro_dir 
                                 + '/' + subject + 
                                 '/rest_label_trial_data_' + band + '_ds.pkl')
    
# calculate long range (s)correlation between pupil and MEG epochs in labels:
fs = 256
band = ''
for subject in subjects:
    sub_pro_dir = MEG_pro_dir + '/' + subject
    pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                                 '/clean_interp_pupil_ds.pkl')
    
    label_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir  + '/' + subject + 
                                 '/rest_label_trial_data_' + band  + '_ds.pkl')
    filename = sub_pro_dir + '/' + epoch_name + '_ds-epo.fif'
    
    events = mne.read_events(filename)
    n_trials = len(events)
    n_times =  np.int(fs * 4 - 1)
    r_pupil_meg = np.zeros((n_labels))
    for l in range(n_labels):
        if l>0: continue;
        x1 = []; x2 = [];
        for trial_number in range( n_trials ):
            sample = int(events[trial_number, 0] * fs / 1200)
            one_trial_pupil = pupil_data[sample:(sample + fs*2)]

            x1.append((label_data[l, trial_number, :]))
            x2.append((one_trial_pupil))
        
        all_pupil = np.array(x2).flatten()
        print(len(all_pupil))
        all_meg = np.array(x1).flatten()
        plt.scatter(all_pupil,all_meg, s = 1)
        p_group = np.digitize(all_pupil, np.percentile(all_pupil, group_percentile))
        p_group = np.digitize(all_pupil, np.linspace(-3.5, -2.5, 10))
        plt.scatter(p_group,all_meg, s = 1)
        mm = [all_meg[p_group == g].mean() for g in range(0,9) ]
        st = [all_meg[p_group == g].std() for g in range(0,9) ]
        plt.errorbar(range(0,9), mm,st)
        r, pval = scipy.stats.pearsonr(x1,x2)
        r_pupil_meg[l] = r
    HLTP_pupil.save(r_pupil_meg, HLTP_pupil.MEG_pro_dir 
                                 + '/' + subject + '/r_pupil_meg_in_label_' + band + '-ds.pkl')
# mean across subjects
r_pupil_meg_all = []
band = 'delta'
for subject in subjects:
    r_pupil_meg_all.append(HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
         + '/' + subject + '/r_pupil_meg_in_label_' + band + '-ds.pkl'))
r_pupil_meg_all = np.array(r_pupil_meg_all)
tval, pval = scipy.stats.ttest_1samp(r_pupil_meg_all, popmean = 0) 
print('significant labels')
sigl = list(np.where(pval < .01)[0])
for l in sigl:
    print(labels[l].name + ' ' + str(r_pupil_meg_all[:, l].mean()) )

# calculate cross-correlation between pupil and MEG epochs in labels:
fs = 256
for subject in subjects:
    sub_pro_dir = MEG_pro_dir + '/' + subject
    pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                                 '/clean_interp_pupil_ds.pkl')
    fc = 5  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    #pupil_data = signal.filtfilt(b, a, pupil_data)

    label_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir  + '/' + subject + 
                                 '/rest_label_trial_data_' + band  + 'ds.pkl')
    
    # continuous cross-correlation - problem because of blinks? 
    #s = label_data.shape
    #cont_label_data = label_data.reshape((s[0],s[1] * s[2]))
    #for l in range(n_labels):
    #    x1 = label_data[l, trial_number, :]
    #    x2 = zscore(pupil_data)
            
    #    cc = np.true_divide(np.correlate(x2, x1, "full"), 
    #                            np.sqrt(np.dot(x1, x1) * np.dot(x2,x2))) 

    filename = sub_pro_dir + '/' + epoch_name + '_ds-epo.fif'
    
    events = mne.read_events(filename)
    n_trials = len(events)
    n_times =  np.int(fs * 4 - 1)
    cc_data = np.zeros((n_labels, n_trials, n_times))
    for l in range(n_labels):
        for trial_number in range( n_trials ):
            sample = int(events[trial_number, 0] * fs / 1200)
            one_trial_pupil = pupil_data[sample:(sample + fs*2)]

            x1 = label_data[l, trial_number, :]
            x2 = zscore(one_trial_pupil)
            
            #x1 = signal.filtfilt(b, a, x1)
            #plt.figure();plt.plot(x1);plt.plot(x2)
            #zeros lag is pearson corrcoef
            cc = np.true_divide(np.correlate(x2, x1, "full"), 
                                np.sqrt(np.dot(x1, x1) * np.dot(x2,x2))) 
            #plt.figure();plt.plot(cc);plt.title(str(trial_number))
            # peak at positive lag = x1(brain) precedes x2
            cc_data[l, trial_number, : ] = cc
    HLTP_pupil.save(cc_data, HLTP_pupil.MEG_pro_dir 
                                 + '/' + subject + '/xcorr_in_label_' + band + '-ds.pkl')
# mean across subjects
fs = 256 
time = np.arange(-2 * fs + 1, 2*fs ) / fs
#time = np.arange(-2 * fs , 2*fs + 1 ) / fs
band = 'beta'
cc_data_all = []
for subject in subjects:
    cc_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                 + '/' + subject + 
                                 '/xcorr_in_label_-ds.pkl')
    cc_data_all.append(np.mean(cc_data, axis = 1))
cc_data_all = np.array(cc_data_all)    
cc_ave = np.mean(cc_data_all, axis = 0)

sig = []
for l in range(n_labels):
    cc_ave_l = cc_data_all[:, l, :].mean(axis = 0)
    cc_err_l = cc_data_all[:, l, :].std(axis = 0)/np.sqrt(19)
    tval, pval = scipy.stats.ttest_1samp(cc_data_all[:, l, :], popmean = 0)  
    sig.append(pval < 0.05)
sig = np.array(sig)
cc_ave_thr = cc_ave.copy()
cc_ave_thr[~sig] = 0
#ss = np.argsort(cc_ave[:, time == time[600]][:,0])
ss = range(44)
lnames = [labels[l].name for l in ss]
fig, ax = plt.subplots(1, 1, figsize = (5.,7))
plt.imshow(cc_ave_thr[ss, :], cmap = 'RdBu_r', aspect = 'auto', alpha = 0.9)
plt.imshow(cc_ave[ss, :], cmap = 'RdBu_r', aspect = 'auto', alpha = 0.3)
plt.xticks(np.linspace(0, 1023, 5), np.linspace(-2, 2, 5))
plt.yticks(np.arange(44), lnames)
fig.savefig(figures_dir + '/hm_xcorr_pupil5hz_rest_' + band + '.png', 
                bbox_inches = 'tight', transparent = True)
  


ax = sns.heatmap(cc_ave[ss, :], vmin=-.05, vmax=.05, cmap = 'RdBu_r')


for l in range(n_labels):
    cc_ave_l = cc_data_all[:, l, :].mean(axis = 0)
    cc_err_l = cc_data_all[:, l, :].std(axis = 0)/np.sqrt(19)
    tval, pval = scipy.stats.ttest_1samp(cc_data_all[:, l, :], popmean = 0)    

    fig, ax = plt.subplots(1, 1, figsize = (4.,2.5)) 
    plt.title(labels[l].name)
    #plt.plot(time, cc_data_all[:, l, :].T, 'gray') 
    plt.plot(time, cc_ave[l,:], 'k') 
    plt.fill_between(time, cc_ave_l-cc_err_l, cc_ave_l+cc_err_l, 
                     facecolor='k', alpha=0.1)
    plt.plot(time[np.where(pval < 0.05)[0]], 
             cc_data_all[:, l, :].mean(axis = 0)[np.where(pval < 0.05)[0]], 'y.')  
    plt.plot(time[np.where(pval < 0.01)[0]], 
             cc_data_all[:, l, :].mean(axis = 0)[np.where(pval < 0.01)[0]], 'r.')  
    plt.ylim([-0.05, 0.05])
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-Correlation (r)')
    plt.plot([0, 0], [-0.1, 0.1], 'k--');plt.plot([-2, 2], [0, 0], 'k--')
    fig.savefig(figures_dir + '/xcorr_pupil5hz_rest_' + band + str(l) + '.png', 
                bbox_inches = 'tight', transparent=True)
  
# plot
maxtime = []    
for l in range(n_labels):
    maxtime = []
    for subject in subjects:
        cc_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                     + '/' + subject + '/xcorr_in_label-ds.pkl')
        #for l in range(1, n_labels):   
            # do here cluster analysis and extract the mean of significant cluster
            # plot distribution across subjects in each area
        tval, pval = scipy.stats.ttest_1samp(cc_data[l, :, : ], popmean = 0)  
        mean_data = np.abs(cc_data[l, :, : ].mean(axis = 0))
        mean_data[pval > 0.05] = 0
        maxtime.append(time[np.argmax(mean_data)])
    plt.figure()
    plt.title(labels[l].name)
    plt.hist(maxtime, bins = np.arange(-2, 2, .1))
    
    tval, pval = scipy.stats.ttest_1samp(cc_data[l, :, : ], popmean = 0)    
    ax = plt.figure()
    plt.title(labels[l].name)
    plt.plot(time, mean_data)  
    plt.plot(time[np.where(pval < 0.01)[0]], 
             cc_data[l, :, :].mean(axis = 0)[np.where(pval < 0.01)[0]], '.')  
    plt.ylim([-0.1, 0.1])
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-Correlation (r)')
    plt.plot([0, 0], [-0.1, 0.1], 'k--');plt.plot([-2, 2], [0, 0], 'k--')
    
ax = plt.figure()
plt.title(labels[l].name)
plt.plot(time, cc_data[l, :, : ].T)     