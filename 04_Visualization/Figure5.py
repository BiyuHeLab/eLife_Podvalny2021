

import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
#import mne
from matplotlib import pyplot as plt
import pandas as pd
def pair_bar_plot(data, chance, tag):
    fig, ax = plt.subplots(1, 1, figsize = (1.4, 2.5))
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions = range(4), patch_artist = True, widths = 0.8,showfliers=False,
             boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    for t in range(4):
        box1['boxes'][t].set( facecolor = [.9,.9,.9], lw=0, zorder=0)
        box1['medians'][t].set( color =  [.4,.4,.4], lw=2, zorder=20)
        plt.plot([t], [data[t]], 'o',
             markerfacecolor='None', color=[.4,.4,.4],
             alpha=0.5, zorder=30)
    plt.plot(range(4), chance, 'dr')

    ax.set_xlim([0., 4.])
    plt.xlabel('Model Type'); plt.ylabel('AUROC')
    plt.ylim([0.2, .8]);plt.xlim([-.45, 3.45])
    fig.savefig(figures_dir + '/prestim_decod_' + tag + '.png',
                bbox_inches = 'tight', transparent=True)

def plot_coef(coefs, model_type):
    for rsn in range(7):
        betas = np.array(coefs)[:, :, (rsn * 5):(rsn + 1)*5].mean(axis = 0).mean(axis = 0)
        errs = np.array(coefs)[:, :, (rsn * 5):(rsn + 1)*5].std(axis = 0)[0]/np.sqrt(24)
        fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
        plt.bar([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'],
                betas, yerr = errs, facecolor = 'w', edgecolor = 'k', capsize = 4)
        plt.ylim([-.4, .4])
        ax.spines["top"].set_visible(False);
        ax.spines["right"].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.ylabel('', fontsize = 14)
        plt.xlabel('Freq. band', fontsize = 14)
        fig.savefig(figures_dir + '/logRcoefs_' + str(rsn) + model_type + '.png', dpi = 800,
                    bbox_inches = 'tight', transparent=True)

    if model_type == 'power_pupil':
        betas = np.array(coefs)[:, :, 35].mean(axis = 0).mean(axis = 0)
        errs = np.array(coefs)[:, :, 35].std(axis = 0)[0]/np.sqrt(24)
        fig, ax = plt.subplots(1, 1,figsize = (.5,2))
        plt.bar(['pupil'],
                    betas, yerr = errs, facecolor = 'w', edgecolor = 'k', capsize = 4)
        plt.ylim([-.4, .4])
        ax.spines["top"].set_visible(False);
        ax.spines["right"].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.ylabel('', fontsize = 14)
        fig.savefig(figures_dir + '/pupil_coef_' + model_type + '.png', dpi = 800,
                        bbox_inches = 'tight', transparent=True)

def plot_coef_table():

    tab_pe = np.array(coefs['power']).mean(axis =0).mean(axis =0 ).reshape((7, 5))
    fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
    plt.imshow(tab_pe, vmin = -.3, vmax = .3, cmap = 'Spectral_r')
    ax.axis('off')

    tab_pe = np.array(coefs['power_pupil']).mean(axis =0).mean(axis =0 )[:35].reshape((7, 5))
    fig, ax = plt.subplots(1, 1,figsize = (1.5,2))
    plt.imshow(tab_pe, vmin = -.3, vmax = .3, cmap = 'Spectral_r')
    ax.axis('off')


def plot_res_power_BHV(savedir):
    ylims = {'HR': 0.02, 'FAR': 0.02,
             'c': 0.05, 'd': 0.05, 'p_corr': 0.02, 'catRT': 0.02}
    ylabels = {'HR': 'Hit Rate', 'FAR': 'FAR',
               'c': 'criterion', 'd': 'sensitivity', 'p_corr': 'p correct',
               'catRT': 'cat. RT'}
    # group_percentile = np.arange(10, 100, 20)
    for bhv_var in ['HR', 'FAR', 'c', 'd', 'p_corr', 'catRT']:
        [table_par_e, table_pvals, table_pvals_corr] = HLTP_pupil.load(HLTP_pupil.result_dir +
                                                                       '/respe_pval_' + bhv_var + '.pkl')

        fig, ax = plt.subplots(1, 1, figsize=(1.5, 2))
        plt.imshow(table_par_e, vmin=-ylims[bhv_var], vmax=ylims[bhv_var],
                   cmap='Spectral_r')
        for (j, i), _ in np.ndenumerate(table_pvals):
            if table_pvals_corr[j, i] < 0.05:  # corrected for mc
                ax.text(i, j, '*', color='w', ha='center', va='center', fontsize=13)
            elif table_pvals[j, i] < 0.05:  # uncorrected
                ax.text(i, j, '+', color='w', ha='center', va='center', fontsize=10)
        ax.axis('off')
        # plt.box(True)
        # ax.spines['top'].set_visible(True);    ax.spines['right'].set_visible(True)
        fig.savefig(savedir + '_betas_' + bhv_var + '.png', dpi=800,
                    bbox_inches='tight', transparent=True)

# FIGURE Power -> Behavior
# Figure 5
savedir = figures_dir + '/res_pwr_bhv_'

plot_res_power_BHV(savedir)
img_type = 'all'
scores, coefs, mean_perm, pop_pval = HLTP_pupil.load(MEG_pro_dir + '/pupil_result/predict_rec_from_' + img_type)
m_types = [ 'power', 'power_pupil', 'power_residual', 'pupil']
data = [scores[t] for t in m_types]

chance = [mean_perm[t] for t in m_types]
pair_bar_plot(data, chance, img_type)
plot_coef(coefs['power_pupil'], 'power_pupil')
plot_coef(coefs['power'], 'power')

plt.bar(range(36), np.array(coefs['power_pupil']).mean(axis = 0).mean(axis = 0))
plt.figure()
plt.bar(range(35), np.array(coefs['power']).mean(axis = 0).mean(axis = 0))