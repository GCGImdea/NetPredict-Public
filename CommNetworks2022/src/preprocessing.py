#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def null_cell_cleaning(data):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1)
    return data

def low_variability_cleaning(data):
    # Remove columns with std = 0
    data = data.loc[:, (data.std(axis=0) != 0).values]
    # Remove columns with IQR = 0
    IQR = []
    for col in data.columns:  
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR.append(Q3 - Q1 != 0)
    data = data.loc[:, IQR]
    return data

def outlier_cleaning(data, row_cols):
    if (row_cols == 'cols'):
        cols_outliers = {}
        for col in data.columns:
            data_nrm = data[col]/data[col].std()
            Q3  = data_nrm.quantile(0.75) - data_nrm.median()
            if (Q3 != 0):
                med = data[col].median()
                dev = data[col] - med
                MAD = (1/Q3) * dev.abs().median()
                if (MAD != 0):
                    cols_outliers[col] = [np.sum(data[col] < med - (3 * MAD)) + np.sum(data[col] > med + (3 * MAD))]
                if (MAD == 0):
                    cols_outliers[col] = 0
            else:
                cols_outliers[col] = 0

        data_tmp = np.vstack(list(cols_outliers.values()))
        data_nrm = data_tmp/np.std(data_tmp)
        Q3  = np.quantile(data_nrm,0.75) - np.median(data_nrm)
        med = np.median(data_tmp)
        dev = np.subtract(data_tmp, med)
        MAD = (1/Q3) * np.median(np.abs(dev))
        data_out = data.drop([k for k,v in cols_outliers.items() if v > med + (3 * MAD)], axis=1)

    
    if(row_cols == 'rows'):
        data1 = data.copy()
        data1["N_outlier"] = 0
        for col in data1.columns:
            if (col != 'N_outlier'):
                data_nrm = data1[col]/data1[col].std()
                Q3 = data_nrm.quantile(0.75) - data_nrm.median()
                if (Q3 != 0):
                    med = data1[col].median()
                    dev = data1[col] - med
                    MAD = (1/Q3) * dev.abs().median()
                    if (MAD != 0):
                        data1.loc[(data1[col]<= med - (3 * MAD)) | (data1[col]>= med + (3 * MAD)), 'N_outlier'] += 1    
        data_out = data1.loc[data1["N_outlier"] == 0].drop("N_outlier", axis=1)
    
    return data_out