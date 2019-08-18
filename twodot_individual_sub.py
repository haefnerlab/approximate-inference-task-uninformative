#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:09:38 2018

@author: maddy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:35:13 2017

@author: Maddox
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:22:42 2017

@author: Maddy
"""

import numpy as np
from expyfun.io import reconstruct_dealer, write_hdf5
from scipy.optimize import fmin
from expyfun.analyze import sigmoid
import matplotlib.pyplot as plt


# %%

def get_per_correct(data):
    data = [np.concatenate((np.invert(d[:, 0]), d[:, 1])) for d in data]
    tot_cor = [np.sum(cc) for cc in data]
    total = [len(cc) for cc in data]
    return [100 * float(t) / tot for t, tot in zip(tot_cor, total)]
    

# Put in the needed path. Current working directory by default    
path = ''
fnames = ['016_2017-08-22 10_29_29.762000', '017_2017-08-24 10_21_08.142000',
          '018_2017-08-25 10_29_01.177000', '019_2017-08-25 12_45_52.729000',
          '022_2017-08-29 15_30_07.197000', '023_2017-08-29 18_53_47.427000',
          '025_2017-08-30 13_06_58.638000',
          '026_2017-08-31 11_57_56.940000', '027_2017-09-01 11_28_05.489000',
          '029_2017-09-04 09_42_01.200000',
          '030_2017-09-05 13_20_42.189000', '031_2017-09-07 12_00_04.559000',
          '032_2017-09-07 17_57_15.647000', '034_2017-09-08 18_08_39.705000',
          '036_2017-09-11 15_49_23.560000', '037_2017-09-12 17_58_53.479000',
          '038_2017-09-13 10_42_06.269000', '039_2017-09-13 13_11_53.674000',
          '042_2017-09-16 11_39_12.726000', '043_2017-09-23 11_13_06.733000']

subjects = ['016', '017', '018', '019', '022', '023', 
            '025', '026', 
            '027', '029', '030', '031', '032', '034', '036', '037',
            '038', '039', '042', '043']

pvals = []
d_prime = []
per_cor = []
diff = []
diff_per = []
responses = []
biases = []

ind = 1
for subject, fname in zip(subjects, fnames):
    print(subject)
    # %% read, reformat data
    td = reconstruct_dealer(path + fname + '.tab')[0]
    
    conditions = [1.25, 2.5, 5, 10, 20]
    
    x = [np.concatenate([-td.trackers.flat[2*ii].x, td.trackers.flat[2*ii +1].x]) for ii in range(len(conditions) * 2)]
    responses.append([np.concatenate([np.expand_dims(np.asarray(td.trackers.flat[2* ii].responses, dtype=bool), 1),
                                 np.expand_dims(np.asarray(td.trackers.flat[2 * ii + 1].responses, dtype=bool), 1)], 1) for ii in range(len(conditions * 2))])
    
    per_cor.append(get_per_correct(responses[-1]))    
    percent = [per_cor[-1][:5], per_cor[-1][5:]]
    
    # %% do fits
    data = np.array(percent) / 100.
    lower = 0.5
    upper = 1
    conditions = 20 * np.log10(conditions)

    def likelihood(params, x, y, lower):
        return np.sum((y - sigmoid(x, lower=lower, upper=params[2], 
                                   midpt=params[0], slope=params[1]))**2)
    def likelihood_2(params, x, y, lower, upper):
        return np.sum((y - sigmoid(x, lower=lower, upper=upper, 
                                   midpt=params[0], slope=params[1]))**2)
    def likelihood_3(params, x, y, upper, slope, lower):
        return np.sum((y - sigmoid(x, lower=lower, upper=upper, 
                                   midpt=params[0], slope=slope))**2)
    x_plot_fit = np.arange(0.01, 40, 0.01)
    
    params = []
    fits = [[],[]]
    pooled = np.array(data).sum(0) / 2.
    
    above = np.where([pp > .75 for pp in pooled])[0]
    if len(above) == 0:
        above = 4
    else:
        above = above[0]
    below = np.where([pp < .75 for pp in pooled])[0]
    if len(below) == 0:
        below = 0
    else:
        below = below[-1] 
    if above == below:
        above += 1
    mid = np.mean([conditions[above], conditions[below]])
    slope = (pooled[above] - pooled[below]) / (conditions[above] - conditions[below])
    params_init = [mid, slope * 2, max(pooled)]
    # do pooled fit
    [ok, _, _, _, w] = fmin(likelihood, params_init, 
                            (conditions, pooled, lower), full_output=True,
                            disp=False)
    if ok[-1] > 1: # repeat if lapse rate > 100%
        [ok, _, _, _, w] = fmin(likelihood_2, params_init[:2], 
                                (conditions, pooled, lower, upper), 
                                full_output=True, disp=False)
        ok = np.concatenate((ok, np.expand_dims(np.array(1), 0)), 0)
    if w:
        print(':(')
    
    # do individual condition fits
    for (p, c, i) in zip(data, ['C0', 'C1'], range(2)):
        mid = [ok[0]]
        [temp, _, _, _, w] = fmin(likelihood_3, mid, 
                                  (conditions, p, ok[2], ok[1], lower), 
                                  full_output=True, disp=False)
        params.append(temp)
        if w:
            print(':\'(')
    thresholds = np.array([np.power(10, m / 20) for m in [params[0][0], params[1][0]]])
    thresholds = np.array([-1. / ok[1] * np.log((ok[2] - .5) / (.75 - .5) - 1) + p[0] for p in params])
    thresholds = np.power(10, thresholds / 20)
    perform_imp = (sigmoid(20 * np.log10(thresholds[0]), lower=.5, upper=ok[2], midpt=params[1][0], slope=ok[1]) - .75) * 100

    # %% save the data
    data = dict(responses=responses[-1],
                thresholds=thresholds,
                ok = ok,
                params=params,
                percent=percent,
                perform_imp = perform_imp)

    write_hdf5((path + 'percent_' + subject), data, overwrite=True)

    # %% plot an example
    if subject=='016':
        from collections import OrderedDict
        from matplotlib.transforms import blended_transform_factory
        from matplotlib import rc
        from matplotlib import rcParams
        
        rcParams['font.sans-serif'] = "Arial"
    
        linestyles = OrderedDict(
            [('densely dotted',      (0, (1, 1))),
             ('densely dashed',      (0, (5, 1))),
        
             ('loosely dashdotted',  (0, (3, 10, 1, 10))),
             ('dashdotted',          (0, (3, 5, 1, 5))),
             ('densely dashdotted',  (0, (3, 1, 1, 1))),
        
             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
        c1 = '#55BBEE'
        c2 = '#852255'
        dpi = 600
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=15)
        ms = 7
        plt.figure(figsize=(8.7 / 2.54, 3.75))
        plt.xlabel(u'Auditory Separation (°)')
        plt.ylabel('% correct')
        plt.xticks(np.power(10, conditions / 20), np.power(10, conditions / 20))
        plt.ylim([45, 100])
        percent = np.array(percent)
        plt.xlim([0, 21])
        plt.gca().set_xticklabels(['1.25 ', '', '5', '10', '20'])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        ## %%
        
        plt.plot(np.power(10, conditions / 20), percent[0], 'o', markersize=ms, c=c1)
        plt.plot(np.power(10, conditions / 20), percent[1], 's', markersize=ms, c=c2, zorder=-10)
        plt.plot(thresholds, [75, 75], 'k', linestyle=linestyles['densely dotted'], zorder=-40, lw=1)
        plt.plot([thresholds[0], thresholds[0]], [75, 81.5], 'k', linestyle=linestyles['densely dashed'], zorder=-41, lw=1)
    
        [plt.plot(x_plot_fit, 100. * sigmoid(20 * np.log10(x_plot_fit), lower=0.5,
                                             upper=ok[2], midpt=p, slope=ok[1]),
                  c=c, lw=1.5, alpha=0.75) for p, c in zip(params, [c1, c2])]
        plt.plot(thresholds[0], 75,'o', c=c1, markersize=ms)
        plt.plot(thresholds[0], 75,'o', c='w', markersize=ms-1.5)
        plt.plot(thresholds[1], 75,'s', c=c2, markersize=ms)
        plt.plot(thresholds[1], 75,'s', c='w', markersize=ms-1.5)
        plt.legend(('Central', 'Matched'), loc='best')#, frameon=False)
        plt.xscale('log')
        plt.xticks([1.25, 2.5, 5, 10, 20], [1.25, 2.5, 5, 10, 20])
        plt.xlim([.5, 25])
        plt.ylim([50, 100])
        plt.yticks([50, 75, 100], [50, 75, 100])
        plt.gca().set_xticklabels(['1.25 ', '2.5', '5', '10', '20'])
        plt.minorticks_off()
        plt.subplots_adjust(left=0.175, right=1, top=0.95, bottom=0.23)
        plt.savefig('single_subject.pdf', dpi=dpi)
        plt.savefig('single_subject.png', dpi=dpi)
