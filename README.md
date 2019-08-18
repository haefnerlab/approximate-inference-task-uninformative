# Task-uninformative visual stimuli improve auditory spatialdiscrimination inconsistent with ideal observer causal inference

This repository contains code for our manuscript [https://www.biorxiv.org/content/10.1101/598425v1]

Python code runs on Python 3 (best installed via Anaconda)
Prerequisite libraries
1) Install expyfun from https://github.com/LABSN/expyfun
2) Install bads from https://github.com/lacerbi/bads

Run twodot_individual_sub.py to preprocess all behavioral data and generate example subject plot
THEN twodot_across_sub.py generates other data plots and does stats -- diff_stats, diff_mean_stats, and pval_binom variables contain the stats info contained in the results section

In order to generate the ideal_observer.mat file containing the ideal observer fits, run the ideal_observer_data.m after uncommenting the save file line
