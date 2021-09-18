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


##### figures for revision

from pymer4.models import Lmer
import pandas as pd
from scipy.stats import zscore
import statsmodels.formula.api as smf
import fig_params
from fig_params import *
def compare_r2py():
    n_rois = 7

    coefs_Q_sqr_R = []
    coefs_Q_sqr_py = []
    coefs_Q_lin_R = []
    coefs_Q_lin_py = []
    coefs_L_R = []
    coefs_L_py = []
    for b in ['task_prestim', 'rest']:
        df = pd.read_pickle(HLTP_pupil.result_dir +
                            '/roi_pwr_and_pupil_blocknorm_' + b + '.pkl')
        df.pupil = zscore(df.pupil) # so we have standardized betas
        df["pupil_sqr"] = np.power(df.pupil, 2)

        # fit the model for each roi and frequency band
        for fband in HLTP_pupil.freq_bands.keys():
            for roi in range(n_rois):
                # fit using standard likelihood so AIC/BIC is comparable
                fit_formula = " ~ pupil_sqr + pupil"
                re_fit_formula = fit_formula + " + 0"
                lmm_Q = smf.mixedlm(fband + str(roi) + fit_formula,
                        df.dropna(), groups = df.dropna()["subject"],
                        re_formula = re_fit_formula)
                mdf_Q = lmm_Q.fit(method = 'Powell', reml = False)

                model = Lmer(fband + str(roi) + ' ~ 1 + pupil_sqr + pupil + (0 + pupil_sqr + pupil|subject)',
                             data = df.dropna())
                model.fit(REML = False, method = 'Powell')
                if mdf_Q.pvalues[1] < 0.01:
                        coefs_Q_sqr_R.append(model.coefs.Estimate[1])
                        coefs_Q_sqr_py.append(mdf_Q.params[1])
                        coefs_Q_lin_R.append(model.coefs.Estimate[2])
                        coefs_Q_lin_py.append(mdf_Q.params[2])

                fit_formula = " ~ pupil"; re_fit_formula = fit_formula + " + 0"
                lmm_L = smf.mixedlm(fband + str(roi) + " ~ pupil",
                                df.dropna(), groups = df.dropna()["subject"],
                                re_formula = re_fit_formula)
                mdf_L = lmm_L.fit(method = 'Powell', reml = False)

                model = Lmer(fband + str(roi)  + ' ~ pupil + 1 + (0 + pupil|subject)', data = df.dropna())
                model.fit(REML=False, method='Powell')

                if mdf_Q.pvalues[1] < 0.01:
                        coefs_L_R.append(model.coefs.Estimate[1])
                        coefs_L_py.append(mdf_L.params[1])

fig, ax = plt.subplots(1, 1, figsize = (3.,3.))
plt.scatter(coefs_L_R, coefs_L_py, color = 'k', alpha = 0.5)
plt.plot([-0.07, 0.07], [-0.07, 0.07], 'k--')
plt.ylabel('statsmodels (Python)'); plt.xlabel('LMER (R)')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
fig.savefig(figures_dir + '/compare_R2Py_L.png', dpi=800, bbox_inches='tight', transparent=True)



fig, ax = plt.subplots(1, 1, figsize = (3.,3.))

plt.scatter(coefs_Q_sqr_R, coefs_Q_sqr_py,  color = 'k', alpha = 0.5)
plt.plot([-0.07, 0.07], [-0.07, 0.07], 'k--')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
plt.ylabel('statsmodels (Python)'); plt.xlabel('LMER (R)')
fig.savefig(figures_dir + '/compare_R2Py_Q_sqr.png', dpi=800, bbox_inches='tight', transparent=True)

fig, ax = plt.subplots(1, 1, figsize = (3.,3.))

plt.scatter(coefs_Q_lin_R, coefs_Q_lin_py,  color = 'k', alpha = 0.5)
plt.plot([-0.07, 0.07], [-0.07, 0.07], 'k--')
plt.ylabel('statsmodels (Python)'); plt.xlabel('LMER (R)')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
fig.savefig(figures_dir + '/compare_R2Py_Q_lin.png', dpi=800, bbox_inches='tight', transparent=True)


fig, ax = plt.subplots(1, 1, figsize = (5.,2.))
plt.hist(diff_sqr, np.arange(-0.003, 0.003, 0.0001), color ='k')
plt.xlim([-0.004, 0.004]); plt.ylim([0, 25])
plt.ylabel('# of models'); plt.xlabel('Difference in P.E.')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
fig.savefig(figures_dir + '/PE_diff_Qsqr.png', dpi=800, bbox_inches='tight', transparent=True)

fig, ax = plt.subplots(1, 1, figsize = (5.,2.))
plt.hist(diff_pupil, np.arange(-0.003, 0.003, 0.0001), color ='k')
plt.xlim([-0.004, 0.004]); plt.ylim([0, 25])
plt.ylabel('# of models'); plt.xlabel('Difference in P.E.')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
fig.savefig(figures_dir + '/PE_diff_Qlin.png', dpi=800, bbox_inches='tight', transparent=True)

fig, ax = plt.subplots(1, 1, figsize = (5.,2.))
plt.hist(diff_lin, np.arange(-0.003, 0.003, 0.0001), color ='k')
plt.xlim([-0.004, 0.004]); plt.ylim([0, 25])
plt.ylabel('# of models'); plt.xlabel('Difference in P.E.')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
fig.savefig(figures_dir + '/PE_diff_Llin.png', dpi=800, bbox_inches='tight', transparent=True)
