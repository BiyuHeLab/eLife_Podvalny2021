import HLTP_pupil
from HLTP_pupil import freq_bands, MEG_pro_dir  # , FS_dir, subjects,
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import bic, aic
import numpy as np
import scipy
from sklearn.model_selection import LeaveOneOut, cross_val_predict, permutation_test_score, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import random
from statsmodels.stats.multitest import multipletests

from sklearn.metrics import roc_auc_score

n_roi = 7

def add_subject_residual_power(bhv_df):
    """
    Add a column of residual power after regressing out the pupil-linked power

    Parameters
    ----------
    bhv_df : TYPE
        Dataframe including single trial behavior, pupil size and power.

    Returns
    -------
    bhv_df : TYPE
        DESCRIPTION.

    """
    for roi in range(n_roi):
        for fband, _ in HLTP_pupil.freq_bands.items():
            DV = fband + str(roi)
            bhv_df[ DV + 'res'] = 0
            for subject in HLTP_pupil.subjects:
                subj_df = bhv_df[bhv_df.subject == subject]
                subj_df["constant"] = 1
                mdf_Q = smf.ols(DV + " ~ np.power(pupil, 2) + pupil + constant",
                        subj_df.dropna()).fit()
                mdf_L = smf.ols(DV + " ~ pupil + constant",
                        subj_df.dropna()).fit()
                if mdf_Q.aic < mdf_L.aic:
                    res_val = mdf_Q.resid.values
                else:
                    res_val = mdf_L.resid.values
                bhv_df.loc[
                    bhv_df.subject == subject,  DV + 'res'] = res_val
    return bhv_df

pupil_pwr_df = pd.read_pickle(HLTP_pupil.result_dir +
                    '/roi_pwr_and_pupiltask_prestim.pkl')
pupil_pwr_df = add_subject_residual_power(pupil_pwr_df)

listRSN = []; list_resRSN = []

for roi in range(n_roi):
        for fband, _ in HLTP_pupil.freq_bands.items():
            listRSN.append(fband + str(roi))
            list_resRSN.append(fband + str(roi) + 'res')

df_name = 'all_subj_bhv_df_w_pupil_power'
bhv_df = pd.read_pickle(HLTP_pupil.result_dir +
                          '/' + df_name + '.pkl')

clf = make_pipeline(StandardScaler(), LogisticRegression(C = 1))
cv = LeaveOneOut()

img_type = 'all'
scores = {'power':[], 'power_pupil':[], 'pupil':[], 'power_residual':[]}
perm_scores = {'power':[], 'power_pupil':[], 'pupil':[], 'power_residual':[]}
coefs = {'power':[], 'power_pupil':[], 'pupil':[], 'power_residual':[]}
for subject in HLTP_pupil.subjects:
    subj_bhv_df = bhv_df[bhv_df.subject == subject]
    y = subj_bhv_df.recognition.values

    subj_df = pupil_pwr_df[pupil_pwr_df.subject == subject]
    X = np.array([subj_df[pwr] for pwr in listRSN])[:, y != 0]
    pupil = subj_df.pupil.values[y != 0]
    X_res = np.array([subj_df[pwr] for pwr in list_resRSN])[:, y != 0]
    y = y[y != 0]

    data_by_model_type = {'power':X.T,
                          'power_pupil':np.hstack((X.T, np.expand_dims(pupil, 1))),
                          'pupil':np.expand_dims(pupil, 1),
                          'power_residual':X_res.T}

    for model_type in data_by_model_type.keys():
        prob = cross_val_predict(clf, data_by_model_type[model_type], y,
                             cv=cv, method = 'predict_proba', n_jobs = 10)
        scores[model_type].append(roc_auc_score(y, prob[:, 1]))
        _, perm_score, _ = permutation_test_score(clf, data_by_model_type[model_type], y,
                                       scoring = 'roc_auc') # no cv here, close enough to loo + we need group pval
        perm_scores[model_type].append(perm_score)
        clf.fit(data_by_model_type[model_type], y)
        coefs[model_type].append(clf.named_steps.logisticregression.coef_)


K = 100

mean_perm = {'power':[], 'power_pupil':[], 'pupil':[], 'power_residual':[]}
pop_pval = {'power':[], 'power_pupil':[], 'pupil':[], 'power_residual':[]}
for model_type in ['power', 'power_pupil', 'pupil', 'power_residual']:
    mean_perm_scores = np.zeros(1000)
    for i in range(1000):
        seq = np.array(random.choices(range(K), k = 24))
        mean_perm_scores[i] = np.mean([np.array(perm_scores[model_type])[s, seq[s]] for s in range(24)])
    pop_pval[model_type] = (1 + np.sum(mean_perm_scores >= np.mean(scores[model_type])))/1001
    mean_perm[model_type] = mean_perm_scores.mean()

scipy.stats.wilcoxon(scores['power'], scores['power_pupil'])
scipy.stats.wilcoxon(scores['power'], scores['pupil'])
multipletests(np.array(list(pop_pval.values())), alpha=0.05, method='fdr_bh')
HLTP_pupil.save([scores, coefs, mean_perm, pop_pval],
                MEG_pro_dir + '/pupil_result/predict_rec_from_' + img_type)





# no difference in decoding accuracy between models excluding and including pupil and all power
# - for all images combined, real images only, scrambled images only
# Conclusion: it seems that pupil state does not add additional information contributing to decoding accuracy
scipy.stats.wilcoxon(score_power, score_power_pupil)
# Significantly reduced model performance for power residuals compared to full power, in all trials!
# Conclusion: pupil-linked power modulation is important in predicting future behavior
scipy.stats.wilcoxon(scores['power'], scores['power_residual'])
#All trials WilcoxonResult(statistic=59.0, pvalue=0.007920503616333008)
#Scram WilcoxonResult(statistic=28.0, pvalue=0.072998046875)
#Real WilcoxonResult(statistic=66.0, pvalue=0.015044927597045898)


