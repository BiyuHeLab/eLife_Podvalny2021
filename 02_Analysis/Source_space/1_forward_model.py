#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:51:55 2017

Compute forward solution, 1st step in source localization

@author: podvalnye
"""
import mne
from  HLTP_pupil import MRI_dir, MEG_pro_dir
import HLTP_pupil

conductivity = (0.3,)  # for single layer, suitable for MEG
subjects = HLTP_pupil.mri_subj['good']
for subject in subjects:
    
    # COMPUTE SOURCE SPACE
    src = mne.setup_source_space(subject, spacing = 'oct6', 
                     subjects_dir = MRI_dir, add_dist = False, n_jobs = -1)
    
    mne.write_source_spaces(MEG_pro_dir + '/' + subject + '/oct6-src.fif', src,
                            overwrite=True)
    
    # COMPUTE BEM MODEL -this depends only on head geometry based on mri 
    model = mne.make_bem_model(subject = subject, ico = 4, 
                               conductivity = conductivity, 
                               subjects_dir = MRI_dir)
    bem = mne.make_bem_solution(model)
    
    mne.write_bem_solution(MEG_pro_dir + '/' + subject + '/bem-sol.fif', bem)
    
# for subjects with bad or without MRI    
subjects = HLTP_pupil.mri_subj['bad'] + HLTP_pupil.mri_subj['no']
template_subj = 'AL'
# I use ALs brain as a template, which looks close enough to fsaverage
# couldn't find fsaverage in right format / figure out the conversion / don't think it matters
for subject in subjects:
    src = mne.setup_source_space(template_subj, spacing = 'oct6', 
                     subjects_dir = MRI_dir, add_dist = False, n_jobs = -1)
    mne.write_source_spaces(MEG_pro_dir + '/' + subject + '/oct6-src.fif', src,
                            overwrite=True)
    model = mne.make_bem_model(subject = template_subj, ico = 4, 
                               conductivity = conductivity, 
                               subjects_dir = MRI_dir)
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(MEG_pro_dir + '/' + subject + '/bem-sol.fif', bem)

# for the two subjects without mri I just copied template trans file
# in their directory, I used the actual trans files for bad mri subjects
    
# COMPUTE FORWARD SOLUTION for HLTP
epoch_name = 'HLTP_raw_stim'
for subject in subjects:   
    sub_pro_dir = HLTP_pupil.MEG_pro_dir + '/' + subject
    sub_raw_dir = HLTP_pupil.MEG_raw_dir + '/' + subject + '/MEG'
    src_fname = sub_pro_dir + '/oct6-src.fif'
    bem_fname = sub_pro_dir + '/bem-sol.fif'
    trans_fname = sub_raw_dir + '/' + subject + '-trans.fif'
    
    # make forward operator
    epochs = mne.read_epochs(MEG_pro_dir + '/' + subject + '/' + epoch_name + 
                             '-epo.fif')
    # Note that epochs file is used only for info about general recording 
    # parameters, such as channels, dev_head_t, nothing of the epochs content
    src = mne.read_source_spaces(src_fname)
    bem = mne.read_bem_solution(bem_fname)
    fwd = mne.make_forward_solution(epochs.info, trans = trans_fname, 
                                    src = src, bem = bem, 
                                    eeg = False,  n_jobs = -1)
    mne.write_forward_solution(sub_pro_dir + '/HLTP_fwd.fif', 
                               fwd, overwrite = True)

    
# Plot visualization of forward solution / make sure worked
for subject in HLTP_pupil.mri_subj['good']:
    mne.viz.plot_bem(subject=subject, subjects_dir=MRI_dir, slices = 90,
                 brain_surfaces='white', orientation='coronal')

