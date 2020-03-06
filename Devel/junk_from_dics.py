#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:54:24 2020

@author: podvae01
"""






# -------------------------------------------------------------------------
# plot fullband pupil x freq matrix
#---------------------------------------------------------------   ---------- 
def get_pupil_groups(epoch_fname, group_percentile):
    # add events indicating pupil size group, move this somewhere else later
    epochs = mne.read_epochs(epoch_fname, preload = True )
    pupil_epochs = epochs._data[:, HLTP_pupil.pupil_chan, :]
    mean_pupil = pupil_epochs.mean(axis = 1)
    perc = np.percentile(mean_pupil, group_percentile)
    p_group = np.digitize(mean_pupil, perc)   
    #p_group = np.digitize(mean_pupil, 
    #                      np.linspace(mean_pupil.min(), 
    #                                 mean_pupil.max()+.001,  21))
    return p_group,  mean_pupil 

group_percentile = np.arange(0., 100., 20);
n_roi = 7

epoch_name = 'task_prestim_ds'
n_freq = 99
power_map = np.zeros( (n_roi, len(group_percentile), n_freq, len(subjects)) )
def ushape(x, a,b,c):
    return a*x**2+b*x+ c

for s, subject in enumerate(subjects):
    sub_pro_dir = MEG_pro_dir + '/' + subject
    epoch_fname = sub_pro_dir + '/' + epoch_name + '-epo.fif'

    p_group, pupil_size = get_pupil_groups(epoch_fname, group_percentile)
    pupil = np.unique(p_group)
    roi_data = HLTP_pupil.load(sub_pro_dir + '/roi_epoch_dics_power_' 
                     + epoch_name)
    
    n_roi, n_freq, n_epoch = roi_data.shape
    
    for roi in range(7):
        for p in pupil:
            power_map[roi, p - 1, :, s] = roi_data[roi, :, p_group == p].mean(axis = 0)
        for f in range(n_freq):
            power_map[roi, :, f, s] = np.log(power_map[roi, :, f, s] / 
                     np.nanmean(power_map[roi, :, f, s]))
   
HLTP_pupil.save(power_map, MEG_pro_dir + 
                    '/pupil_result/group_roi_power_map_7nets_' + epoch_name)

for roi in range(n_roi):
    fig, ax = plt.subplots(figsize = [2., 3.]);  
    tstt, pval=stats.ttest_1samp(power_map[roi, :, :, :], 
                                 popmean = 0, axis= -1 )    
    sig = masked_array(tstt, pval<0.05)
    nsig = masked_array(tstt, pval>=0.05)
    plt.imshow(tstt.T, interpolation = 'hamming',
                   vmin = -3., vmax = 3., cmap = 'inferno', 
                  extent = [-10, 10, 100, 1]);     
    #plt.imshow(sig.T, interpolation = 'hamming',
    #               vmin = -3., vmax = 3., cmap = 'RdYlBu_r', 
    #              extent = [-10, 10, 100, 1]);
    plt.colorbar()      
    plt.yscale('log'); #plt.xscale('log'); 
    plt.xlabel('Pupil size (a.u.)')
    plt.ylabel('Frequency (Hz)')
    fig.savefig(figures_dir + '/power_map_7by_pupil_size' 
                + str(roi) + '.png', bbox_inches = 'tight', 
                dpi = 800, transparent = True)
    
           
for roi in range(n_roi):
    for s in range(8):    
        plt.figure(figsize = (3,3))
        for band in freq_bands:
            mpo = power_map[roi, :, freq_bands[band][0]:(freq_bands[band][1]+1), s].mean(axis = -1)
            plt.plot(mpo, label = band)
    plt.legend()    
    plt.xlabel('Pupil size (a.u.)')   
    plt.ylabel('power')

for roi in range(n_roi):
    plt.figure(figsize = (3,3))   
    plt.imshow(np.nanmean(power_map[roi, :, :, :], axis = -1).T / 
               np.std(power_map[roi, :, :, :], axis = -1).T, 
               interpolation = 'gaussian',
                   vmin = -.8, vmax = .8, cmap = 'RdYlBu_r')
    plt.yscale('log')
    plt.xlabel('Pupil size (a.u.)')
    plt.ylabel('Frequency (Hz)')

def get_data_in_rois():
    subjects_dir = '/isilon/LFMI/VMdrive/Ella/mne_data/MNE-sample-data/subjects/'
    hs = 'rh'

    labels = mne.read_labels_from_annot(
        'fsaverage', 'HCPMMP1_combined', hs, subjects_dir=subjects_dir)

    rois = [label.name for label in labels];rois = rois[1:]
    roi_idx = []
    for i, r in enumerate(rois):
        roi_idx.append(np.where(np.array([ l.name == rois[i] 
            for l in labels ]))[0][0])


#----------------GLM to predict behavior --------------------------------------

all_betas = np.zeros( (7, 8, len(subjects)) )
for roi in range(7):
    betas = []
    for s, subject in enumerate(subjects):  
        sub_pro_dir = MEG_pro_dir + '/' + subject
        epoch_fname = sub_pro_dir + '/' + epoch_name + '-epo.fif'
    
        p_group, pupil_size = get_pupil_groups(epoch_fname, group_percentile)
        roi_data = HLTP_pupil.load(sub_pro_dir + '/roi_epoch_dics_power_' 
                         + epoch_name)
        pupil_size = zscore(pupil_size)
        power = np.log(np.array([roi_data[roi, (freq_bands[band][0]-1):freq_bands[band][1], :
            ].mean(axis = 0) for band in freq_bands]))
        power = zscore(power, axis = 1)
        Y = bhv_dataframe[bhv_dataframe.subject == subject].correct.values
        valid = np.ones(len(Y)).astype('bool')   
        #valid = bhv_dataframe[bhv_dataframe.subject == subject].real_img.values
     
        if len(pupil_size) < len(Y): Y = Y[:len(pupil_size)]; valid = valid[:len(pupil_size)]
        X = np.stack([
                
                power[0,:],
                power[1,:],
                power[2,:],
                power[0,:] * pupil_size,
                power[1,:] * pupil_size,
                power[2,:] * pupil_size,
                pupil_size]).T

        model = sm.GLM(Y[valid], 
                       sm.add_constant(zscore(X[valid, :].squeeze(), axis = 0)), 
                       family = sm.families.Binomial())
        
        res = model.fit()
        beta = res.params
        
        #plt.figure();plt.bar(range(len(beta)), beta, yerr = res.bse)
        all_betas[roi, :, s]= beta
for roi in range(7):
    dd = all_betas[roi, :, :]
    fig, ax = plt.subplots(figsize = [2., 2.]);
    plt.bar(range(len(beta)), dd.mean( axis = -1), 
            yerr = dd.std(axis = -1) / np.sqrt(len(subjects)))
    plt.ylim([-.5, .5])
    tstas, pval = scipy.stats.ttest_1samp(dd, popmean = 0, axis = 1)
    print(pval < 0.05)
    plt.ylabel("param. est.")
    fig.savefig(figures_dir + '/GLM_power_pupil_' + str(roi) + '.png', 
                               bbox_inches = 'tight',  dpi = 800, transparent = True)
 
# mean roi
    dd = all_betas.mean(axis = 0)
    fig, ax = plt.subplots(figsize = [2., 2.]);
    plt.bar(range(len(beta)), dd.mean( axis = -1), 
            yerr = dd.std(axis = -1) / np.sqrt(len(subjects)))
    plt.ylim([-.5, .5])
    tstas, pval = scipy.stats.ttest_1samp(dd, popmean = 0, axis = 1)
    print(pval < 0.05)
    plt.ylabel("param. est.")
    fig.savefig(figures_dir + '/GLM_power_pupil.png', 
                   bbox_inches = 'tight',  dpi = 800, transparent = True)

# PLOTTING - TO BE MOVED   

# plot for each frequency band how power changes with pupil in networks

brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir= FS_dir,
              cortex='low_contrast', background='white', size=(1000, 1000))
brain.add_annotation('Yeo2011_7Networks_N1000')

brain.save_image(figures_dir + '/test.png')

def prepare_data_for_anova(stcs):
    data = []
    n_subjects = len(norm_stcs)
    for fband, band in enumerate(freq_bands.keys()):
        #subjects X pupil_group X frequency_band X voxels 
        for grp in range(1,6):
            data_grp = np.zeros((n_subjects, 20484))
            for s in range(n_subjects):
                data_grp[s, :] = stcs[s][str(grp)]._data[:, fband]
            data.append(data_grp)
    return data

def two_way_ANOVA_pupil_freqband_interaction(data):
    mne_dir = '/isilon/LFMI/VMdrive/Ella/mne_data/MNE-sample-data'
    src_fname = mne_dir + '/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
    src = mne.read_source_spaces(src_fname)
    connectivity = mne.spatial_src_connectivity(src)
    effects =  'A:B'

    factor_levels = [5, 5]# two factors : pupil group x frequency bands
    crit_voxel_alpha = 0.05
    crit_clust_alpha = 0.05
    f_thresh = mne.stats.f_threshold_mway_rm(19, factor_levels, 
                         effects, crit_voxel_alpha)#, 'A', 'B'
    
    def stat_fun(*args):
            return mne.stats.f_mway_rm(np.swapaxes(
                    np.array(args), 0, 1), factor_levels, effects = effects, 
                return_pvals=False)[0]

    F_obs, clusters, cluster_p_values, H0  = \
            mne.stats.permutation_cluster_test(data,
                             connectivity = connectivity.astype('int'),
                             n_jobs = -1, tail = 0,
                             threshold = f_thresh, stat_fun=stat_fun,
                             n_permutations = 1000) 
    good_cluster_inds = np.where(cluster_p_values < crit_clust_alpha)[0]             
    F_val =  np.zeros(F_obs.shape)
    for i in good_cluster_inds:
        F_val[clusters[i]] = F_obs[clusters[i]] 
    return F_val


# ANOVA
data = prepare_data_for_anova(raw_stcs)
F_val = two_way_ANOVA_pupil_freqband_interaction(data)    
        
#band_stc._data = np.expand_dims(F_val, axis = 1)
#band_stc.save(results_dir + '/' + epoch_name + 
#                      '_fbands_src_ANOVA_interaction', verbose = 5)    
    
band_stc = mne.read_source_estimate(results_dir + '/' + epoch_name + 
                  '_fbands_src_ANOVA_interaction')
fmax = band_stc._data.max()
fig = mlab.figure(size=(300, 300))
band_stc.plot(subjects_dir=FS_dir,
                subject='fsaverage', figure = fig, 
                hemi='both', transparent = True, background = 'white',
                time_label='', views='lateral', alpha = 0.95,
                clim=dict(kind='value', lims=[0.0, 3, 5]))
time.sleep(2)
mlab.savefig(figures_dir + '/MEG_' + epoch_name + 
             '_fbands_src_lateral_ANOVA_interaction.png')    

#------1WAY repeated measures ANOVA with pupil group as factor---
factor_levels = [5]# 
effects = ['A']            
crit_alpha = 0.05
f_thresh = mne.stats.f_threshold_mway_rm(23, factor_levels, 
                                         effects , crit_alpha)
def stat_fun(*args):
        return mne.stats.f_mway_rm(np.swapaxes(
                np.array(args), 0, 1), factor_levels, effects = effects, 
            return_pvals=False)[0]

for fband, band in enumerate(freq_bands.keys()):
    #subjects X pupil_group X voxels 
    data = []

    band_stc= stc_sub_mean[grp].copy().crop(fband, fband).mean() 
    for grp in range(1,6):
        data_grp = np.zeros((n_subjects, 20484))
        for s in range(n_subjects):
            data_grp[s, :] = stcs[str(grp)][s]._data[:, fband]
        data.append(data_grp)

    T_obs, clusters, cluster_p_values, H0 = clu = \
            mne.stats.permutation_cluster_test(data,
                             connectivity = connectivity.astype('int'),
                             n_jobs = -1, tail = 0,
                             threshold = f_thresh, stat_fun=stat_fun,
                             n_permutations = 5000)        
    
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]             
    F_val =  np.zeros(T_obs.shape)
    for i in good_cluster_inds:
        F_val[clusters[i]] = T_obs[clusters[i]] 
    band_stc._data = np.expand_dims( F_val, axis = 1)
    band_stc.save(results_dir + '/' +  epoch_name + '_src_1wANOVA_'  + band, 
                  verbose = 5)
    
for fband, band in enumerate(freq_bands.keys()):

    band_stc = mne.read_source_estimate(results_dir + '/' +  epoch_name + 
                                        '_src_1wANOVA_'  + band)
    fmax = band_stc._data.max()
    fig = mlab.figure(size=(300, 300))
    #fig.scene.off_screen_rendering = True 
    band_stc.plot(
                    subjects_dir=FS_dir, title = band ,
                    subject='fsaverage', figure = fig, 
                    hemi='both', transparent = True, background = 'white',
                    time_label='', views='lateral', alpha = 0.95,
                    clim=dict(kind='value', lims=[0.01, 5, 15]))
    fig.scene.off_screen_rendering = True
    mlab.savefig(figures_dir + '/MEG_' + epoch_name + 
                 '_src_lateral_1wANOVA_' + band+ '.png')




hs = 'rh'
#labels = mne.read_labels_from_annot(
#    'fsaverage', 'aparc', hs, subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot(
    'fsaverage', 'Yeo2011_7Networks_N1000', hs, subjects_dir=FS_dir)

#rois = ['lateraloccipital-' + hs, 'inferiortemporal-' + hs,  'superiorparietal-rh']#,
#        #'fusiform-' + hs, 'parahippocampal-' + hs,'pericalcarine-rh']

#rois = ['frontalpole-' + hs, 'insula-' + hs, 'lateralorbitofrontal-rh', 
#        'medialorbitofrontal-rh', 'superiorfrontal-rh', 'temporalpole-rh']
rois = [label.name for label in labels];rois = rois[:-1]
roi_idx = []
for i, r in enumerate(rois):
    roi_idx.append(np.where(np.array([ l.name == rois[i] 
        for l in labels ]))[0][0])
from matplotlib import cm
stcs = norm_stcs
for i in roi_idx:
    data = np.zeros(shape = (5,5,19))
    for fband, band in enumerate(freq_bands.keys()):
        for grp in range(1,6):
            
            data_grp = np.zeros((n_subjects, len(
                    stcs[s][str(grp)].in_label(labels[i]).data[:, fband])))#20484
            for s in range(n_subjects):
                data_grp[s, :] = stcs[s][str(grp)].in_label(labels[i]).data[:, fband]
                #data_grp[s, :] = stcs[str(grp)][s]._data[:, fband]
            data[fband, grp - 1, :] = (data_grp.mean(axis = -1))
            #F, P = f_mway_rm(data[4,:,:].T, factor_levels = [5])
    
    c=cm.cool(np.linspace(0,1,6))
    fig, ax = plt.subplots(figsize = [3,3]);  
    for grp in range(5):
        plt.errorbar(range(5), np.expand_dims(data[:,grp,:].mean(axis = -1), axis = 1), 
                     yerr = np.expand_dims(data[:,grp,:].std(axis = -1), axis = 1)/np.sqrt(19), 
                     color = c[grp], label = grp);
    plt.ylabel('relative power (dB)')
    plt.xlabel('Freq. band')
    #plt.legend(loc = 'lower right')
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xticks(range(5),list(freq_bands.keys()))
    #plt.ylim([-0.1, 0.1])
    plt.title(labels[i].name)
    fig.savefig(figures_dir + '/MEG_' + epoch_name + str(i) + '_by_pupil_size.png', 
                           bbox_inches = 'tight',  dpi = 800, transparent = True)

for fband, band in enumerate(freq_bands.keys()):
    fig = plt.figure(figsize = (3,3));
    for i in roi_idx:
        lbl_stc = []
        for grp in range(1,6):
            band_stc= stc_sub_mean[grp].copy().crop(fband, fband).mean() 
            lbl_stc.append(band_stc.in_label(labels[i]).data.mean())
        plt.plot(lbl_stc, 'o--', label = labels[i].name, color = labels[i].color, 
                 alpha = 0.9);#plt.title(band)
    #plt.legend(loc = 'lower right')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.ylabel('relative power (dB)')
    plt.xlabel('pupil size')
    fig.savefig(figures_dir + '/MEG_' + epoch_name + band + hs + '.png', 
                       bbox_inches = 'tight',  dpi = 800, transparent = True)

brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=FS_dir,
               cortex='low_contrast', background='white', size=(800, 600))
brain.add_data(F_val)
brain.add_annotation('HCPMMP1_combined')