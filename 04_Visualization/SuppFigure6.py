import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/podvae01/python/HLTP_Pupil/02_Analysis/Sensor_space')
from Pupil_fast_events_related_response import get_pupil_events
block = 'rest01'
con_duration = []; dil_duration = []
con_med = []; dil_med = []
for subject in HLTP_pupil.subjects:
        pupil_events, event_code, _ = get_pupil_events(block, subject)
        pupil_events_in_ms = (pupil_events / HLTP_pupil.resamp_fs) * 1000
        dil_duration.append(np.diff(pupil_events_in_ms[event_code < 0]))
        con_duration.append(np.diff(pupil_events_in_ms[event_code > 0]))
        dil_med.append(np.median(np.diff(pupil_events_in_ms[event_code < 0])))
        con_med.append(np.median(np.diff(pupil_events_in_ms[event_code > 0])))
#plot this on reversed axis like in Joshi et al
fig, ax = plt.subplots(1, 1, figsize = (5.,2.))
plt.hist(np.concatenate(con_duration)/1000, 150, facecolor = 'k')
plt.xlim([0, 6]); plt.ylim([0, 800])
plt.xlabel('inter-event interval (s)')
plt.ylabel('samples')
plt.scatter(np.array(con_med)/1000, np.ones(len(dil_med)) * 750, alpha = 0.2, marker = 'd', c = 'k', edgecolors='k')
fig.savefig(fig_params.figures_dir + '/event_duration_hist_con.png', bbox_inches = 'tight', transparent = True)

fig, ax = plt.subplots(1, 1, figsize = (5.,2.))
plt.hist(np.concatenate(dil_duration)/1000, 150, facecolor = 'k')
plt.xlim([0, 6]); plt.ylim([0, 800])
plt.scatter(np.array(dil_med)/1000, np.ones(len(dil_med)) * 750, alpha = 0.2, marker = 'd', c = 'k', edgecolors='k')
plt.xlabel('inter-event interval (s)')
plt.ylabel('samples')
fig.savefig(fig_params.figures_dir + '/event_duration_hist_dil.png', bbox_inches = 'tight', transparent = True)
