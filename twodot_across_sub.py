#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:05:38 2017

@author: maddy
"""

import numpy as np
import matplotlib.pyplot as plt
from expyfun.io import read_hdf5, write_hdf5
from expyfun.analyze import sigmoid
from scipy.stats import binom, ttest_1samp
from scipy.io import loadmat
import seaborn.apionly as sns
from pandas import DataFrame


plt.rc('font', size=8)
plt.rc('axes', titlesize=8)

xlabels = ['1.25 ', '', '5', '10', '20']

# Put in the needed path. Current working directory by default
path = ''

# %% Read in and plot the model fits with the data

data = loadmat('ideal_observer.mat')
raw = 100 * data['data_raw']
fits = 100 * data['fits']
raw_x = data['data_x'].squeeze()
fits_x = data['fits_x'].squeeze()

plt.style.use('default') #switches back to matplotlib style
from collections import OrderedDict
from matplotlib import rcParams

rcParams['font.sans-serif'] = "Arial"
rcParams['font.size'] = 8
plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
linestyles = OrderedDict(
    [('densely dotted',      (0, (3, 3))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
c1 = '#55BBEE'
c2 = '#852255'
ms=3

plt.figure(figsize=(8.7 / 2.54, 1.75))
raw_mean = raw.mean(0)
raw_mean = (100 - np.fliplr(raw_mean[:, :5]) + raw_mean[:, 5:]) / 2.
raw_x_half = raw_x[5:] * 2.
fits_x_half = fits_x[101:] * 2
fits_half = ((100 - np.flip(fits.mean(0)[0][:100], 0) + fits.mean(0)[0][101:]) 
             / 2)
[plt.plot(raw_x_half+0.05 * np.random.rand(), m, mark, c=c, ms=ms, 
          zorder=z) for m,  mark, c, z in zip(raw_mean, ['o', 's'], [c1, c2], 
                                              [0, -10])]
plt.plot(fits_x_half, fits_half, c1, lw=1)
plt.plot(fits_x_half, fits_half, linestyle=linestyles['densely dotted'], 
         lw=2, c=c2)

plt.ylabel('Percent Response Right')
plt.xlabel(u'Auditory Separation (°)')
plt.legend(('Central', 'Matched'))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([1.25, 2.5, 5, 10, 20], [1.25, 2.5, 5, 10, 20])
plt.gca().set_xticklabels(xlabels)
plt.ylim([50, 100])
plt.xlim([0, 22])
plt.tight_layout()

plt.savefig('model_fits.pdf', dpi=600)

# %% Reformat arrays for easy plotting

raw_diff = np.diff(raw, 1, 1).squeeze()
raw_diff[:, :5] *= -1
fits_diff = np.diff(fits, 1, 1).squeeze()
fits_diff[:, :100] *= -1
raw_x_half = 2 * raw_x[-5:]
fits_x_half = 2 * fits_x[-100:]
raw_half = 0.5 * (np.fliplr(raw_diff[:, :5]) + raw_diff[:, -5:])
fits_half = 0.5 * (np.fliplr(fits_diff[:, :100]) + fits_diff[:, -100:])

# %% Plot difference in two conditions
rcParams['font.sans-serif'] = "Arial"

c3 = '#332288'
ms = 5
plt.figure(figsize=(8.7 / 2.54, 2.5))
angles = 20 * [1.25, 2.5, 5, 10, 20]
angles += list(np.arange(1.25, 30, 1.25))
angles += 20 * [30]
effect = list(raw_half.ravel()) + 23 * [None] + list(raw_half.mean(1))
data = DataFrame(data={'angles': angles, 'effect': effect})
sns.swarmplot('angles', 'effect', data=data, palette=[c3], marker='^', size=3)
sns.pointplot('angles', 'effect', data=data, color=c3, join=False, markers ='^', scale=.1, capsize=.6, errwidth=1)
plt.plot([0, 1, 3, 7, 15], raw_half.mean(0), marker='^', mfc='w', mec=c3,
         markersize=ms, lw=0, zorder=100)
plt.plot([23], raw_half.mean(0).mean(0), marker='^', mfc='w', mec=c3,
         markersize=ms, lw=0, zorder=100)
plt.xticks([0, 1, 3, 7, 15, 23], [1.25, 2.5, 5, 10, 20, 25])
plt.gca().set_xticklabels(xlabels + ['Mean'])
plt.xlabel(u'Auditory Separation (°)')
plt.ylabel('Performance Improvement \n (% correct)')

plt.xlim([-1, 25])
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.subplots_adjust(left=0.175, right=1, top=0.95, bottom=0.2)
plt.savefig(path + 'data_diff.pdf', dpi=600)
plt.plot([-1, 32], [0, 0], 'k', lw=1)
plt.xlim([-1, 25])

plt.savefig(path + 'data_diff_model.pdf', dpi=600)

# %% Read in data from single subject analysis file

subjects = ['016', '017','018', '019', '022', '023', '025', '026', '027',
            '029', '030', '031', '032', '034', '036', '037', '038', '039', '042',
            '043']
x_plot_fit = np.arange(0.01, 30, 0.01)
conditions = [1.25, 2.5, 5, 10, 20]
thresholds = np.empty((len(subjects), 2))
slopes = np.empty((len(subjects), 2))
per_cor = np.empty((len(subjects), 2, 5))
curves = np.empty((len(subjects), 2, len(x_plot_fit)))
curves_resamp = np.empty((len(subjects), 1000, 3999))
improvements = np.empty((len(subjects), 5))
sems = np.empty((len(subjects), 2))
pvals = np.zeros((len(subjects),))
lapse = np.empty((len(subjects), 1))
thresh = np.empty((len(subjects), 2))
slope = np.empty((len(subjects), 1))
perform_imp = np.empty((len(subjects), 1))
data_resamp = np.empty((len(subjects), 1000, 2, 5))
responses = []
biases = []
d_prime = np.empty((len(subjects), 5))
stem_perimp = np.empty((len(subjects), 1000, 2))
stem_thimp = np.empty((len(subjects), 1000, 2))

for i, s in enumerate(subjects):
    data = read_hdf5(path + 'percent_' + s)
    responses.append(data['responses'])
    thresholds[i] = data['thresholds']
    per_cor[i] = data['percent']
    ok = data['ok']
    params = data['params']
    curves[i] = [sigmoid(20 * np.log10(x_plot_fit), lower=0.5, upper=ok[2],
                         midpt=p[0], slope=ok[1]) for p in params]
    perform_imp[i] = data['perform_imp']
    lapse[i] = [ok[2]]
    slope[i] = [ok[1]]
    thresh[i] = [params[0], params[1]]

percent = np.mean(per_cor, 0)

inds = np.argsort(thresholds[:, 0])

# %% Plot improvement in threshold at central threshold

dpi_fig = 200
dpi = 600

ms = 7
threshold = np.mean(thresholds, 0)

for i in range(5):
    points_co = [per[0][i] for per in per_cor]
    points_ma = [per[1][i] for per in per_cor]

ms = 3
rcParams['font.sans-serif'] = "Arial"
c='k'
figsize = (4.1/2.54, 2)
plt.figure(figsize=figsize, dpi=dpi_fig)
plt.plot(thresholds[:, 0], -np.diff(thresholds, 1, -1), 
         marker='^', mfc='w', mec=c3, markersize=ms, lw=0, zorder=100)
plt.plot([0, 25], [0, 0], c='k', lw=1, zorder=-100)
sns.regplot(thresholds[:, 0], np.squeeze(-np.diff(thresholds, 1, -1)), ci=95, marker='', color=c3, line_kws={'linewidth': 1})
plt.xlabel(u'Central threshold (°)', fontname = "Arial")
plt.ylabel(u'Threshold improvement \n (°)')

plt.xticks([1.25, 2.5, 5, 10, 20], [1.25, 2.5, 5, 10, 20])
plt.gca().set_xticklabels(xlabels)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
th_imp= np.squeeze(-np.diff(thresholds, 1, -1))
plt.xlim([0,21.25])
plt.ylim([-3, 10])
plt.subplots_adjust(left=0.4, right=.97, top=0.95, bottom=0.2)

plt.savefig(path + 'stem_deg.pdf', dpi=600)
# %% plot performance improvement at central threshold
figsize = (.7/2.54, 2)
plt.figure(figsize=figsize, dpi=dpi_fig)
from scipy.stats import gaussian_kde
kernel = gaussian_kde(th_imp, 0.3)
plt.plot(20 * kernel(np.arange(-2.5, 10, .01))[kernel(np.arange(-2.5, 10, .01)) > 0.005], np.arange(-2.5, 10, .01)[kernel(np.arange(-2.5, 10, .01)) > 0.005], color=c3, lw=1)
plt.fill_betweenx(np.arange(-2.5, 10, .01)[kernel(np.arange(-2.5, 10, .01)) > 0.005], 20 * kernel(np.arange(-2.5, 10, .01))[kernel(np.arange(-2.5, 10, .01)) > 0.005], color=c3, lw=0)
plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.2)
plt.xlim([0,15])
plt.ylim([-3, 10])
plt.gca().xaxis.set_visible(False)
plt.gca().yaxis.set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.savefig(path + 'stem_deg_marginal.pdf', dpi=600)

plt.figure(figsize=(4.1 / 2.54, 2), dpi=dpi_fig)
plt.plot(thresholds[:, 0], perform_imp, marker='^', mfc='w', mec=c3,
         markersize=ms, lw=0)
plt.plot([0, 25], [0, 0], c='k', lw=1, zorder=-100)
plt.xticks([1.25, 2.5, 5, 10, 20], [1.25, 2.5, 5, 10, 20])
plt.gca().set_xticklabels(xlabels)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim([0,21.25])
plt.xlabel(u'Central threshold (°)')
plt.ylabel('Performance improvement \n (% correct)')
plt.subplots_adjust(left=0.4, right=.97, top=0.95, bottom=0.2)
plt.savefig(path + 'stem_correct.pdf', dpi=600)

improvements = -np.diff(thresholds, 1, -1)


# %% some stats

ax = plt.figure(figsize=(4.2, 2.8))
dist = np.mean(np.diff(per_cor, 1, -2), -1).squeeze()
inds = np.argsort(dist)
write_hdf5('effectsize.hdf5', dist, overwrite=True)


plt.plot(np.arange(20) + 1, dist[inds], '^', c='C2', ms=7)

plt.plot(np.arange(20) + 1, perform_imp[inds], 'd', mec='C2', mfc='w', ms=7)

plt.errorbar(23.6, dist.mean(), dist.std() / np.sqrt(20), fmt='^', color='C2', 
             ms=10, lw=1.5, capsize=3)
plt.errorbar(24.4, perform_imp.mean(), perform_imp.std() / np.sqrt(20), 
             fmt='d', mec='C2', mfc='w', color='C2', ms=10, lw=1.5, capsize=3)

plt.plot([0, 28], [0,0], 'k--', zorder=-5)
plt.xlim([0, 25])
plt.xlabel('Subjects')
plt.ylabel('Improvement \n (percent correct)')
ttest_1samp(dist, 0)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24])
plt.gca().set_xticklabels(['','','','','','','','','','','','','','','','','','','','','Mean'])
plt.yticks([-8, -4, 0, 4, 8])
plt.ylim([-12, 12])
plt.show()
plt.gca().yaxis.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(('Average p=0.007', 'Threshold p=0.03'))
plt.tight_layout()
plt.savefig('stats.pdf', dpi=300)

diff_stats = [ttest_1samp(raw_half[:, i], 0) for i in np.arange(5)]
diff_mean_stats = ttest_1samp(raw_half.mean(1), 0)
positive = sum([imp > 0 for imp in np.mean(improvements, 1)])
probs = binom(20, 0.5)
pval_binom = 1 - probs.cdf(positive) 
