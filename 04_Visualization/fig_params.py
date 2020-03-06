#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:51:50 2020

@author: podvae01
"""

import sys
sys.path.append('../')
from HLTP_pupil import MEG_pro_dir
import numpy as np
from matplotlib import rcParams

results_dir = MEG_pro_dir + '/results'
figures_dir = MEG_pro_dir  +'/_figures'

colors = {'Rec':np.array([255, 221, 140]) / 255., 
          'Unr':np.array([164, 210, 255]) / 255., 
          'All':np.array([174, 236, 131]) / 255.,
          'RecD':np.array([255, 147, 0]) / 255., 
          'UnrD':np.array([4, 51, 255]) / 255., 
          'AllL':np.array([195, 250, 160]) / 255.,
          'AllD':np.array([127, 202, 96]) / 255.,
          'AllDD':np.array([53, 120, 33]) / 255.,
          'sens':np.array([244, 170, 59]) / 255.,
          'bias':np.array([130, 65, 252]) / 255.    }

fig_width = 7  # width in inches
fig_height = 4.2  # height in inches
fig_size =  [fig_width, fig_height]
params = {    
          'axes.spines.right': False,
          'axes.spines.top': False,
          
          'figure.figsize': fig_size,
          
          'ytick.major.size': 8,      # major tick size in points
          'xtick.major.size': 8,    # major tick size in points
              
          'lines.markersize': 6.0,
          # font size
          'axes.labelsize': 14,
          'axes.titlesize': 14,     
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'font.size': 12,

          # linewidth
          'axes.linewidth': 1.3,
          'patch.linewidth': 1.3,
          
          'ytick.major.width': 1.3,
          'xtick.major.width': 1.3,
          'savefig.dpi' : 800
          }
rcParams.update(params)
rcParams['font.sans-serif'] = 'Helvetica'