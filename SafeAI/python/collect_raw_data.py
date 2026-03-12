# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

import sys
sys.path.append('attribute_name_files')
from nokia_data_attributes import attributes_throughput

# dataset   = 'Nk1'
dataset   = 'Nk2'


if (dataset=='Nk1'):
    file_name = 'data/IMDEA.Umlaut.Q1.single.row.csv'
    test_type = 'HTTP LIVE PAGE DL'
    
    print('-----------------------------------------------------------------')
    print('LOADING DATASET')
    data = pd.read_csv(file_name, 
                        encoding = "ISO-8859-1", 
                        sep=";", 
                        decimal=',',
                        low_memory=False)
    data = data[data['Test.Type'] == test_type]
    # data = data[data['Test.Type'] == 'HTTP FILE DL']
    # data = data[data['Test.Type'] == 'HTTP LIVE PAGE DL']
    data = data[data['Test.Qualifier'] == 'QUALIFIED']
    data = data.loc[~data['Test.Throughput.kbit.s'].isna()]
    data = data.loc[data['Test.Throughput.kbit.s'] > 0]
    data_tech = data['Technology']
    data = data[attributes_throughput]
    data = data.select_dtypes(['number'])
    
    # kpi_aux = data['Test.Throughput.kbit.s']
    # data = data[data.columns[data.columns.str.contains('.a2b')|
    #                          data.columns.str.contains('.b2a')|
    #                          data.columns.str.contains('_min')|
    #                          data.columns.str.contains('_25.')|
    #                          data.columns.str.contains('_50.')|
    #                          data.columns.str.contains('_75.')|
    #                          data.columns.str.contains('_max')|
    #                          data.columns.str.contains('_avg')|
    #                          data.columns.str.contains('_sum')]]
    
    # data = data[data.columns[~data.columns.str.contains('VolStep')]]
    # data = data[data.columns[~data.columns.str.contains('TimeStep')]]
    # data = data[data.columns[~data.columns.str.contains('FirstSec')]]
    # data = data[data.columns[~data.columns.str.contains('FirstMB')]]
    
    print('-----------------------------------------------------------------')
    print('Before removing duplicates\t Size:(%d, %d)'%(data.shape[0], data.shape[1]))
    data = data.T.drop_duplicates().T
    print('After removing duplicates\t Size:(%d, %d)'%(data.shape[0], data.shape[1]))
    print('-----------------------------------------------------------------')
    print('Categorical features \t Length:(%d)'%(data_tech.shape[0]))
    print('-----------------------------------------------------------------')
    
    data.columns= data.columns.str.lower()
    original_feat = list(data.columns)
    for i in range(len(original_feat)):
        data.rename(columns={original_feat[i]: original_feat[i].replace('_','.')}, inplace=True)
        data.rename(columns={original_feat[i]: original_feat[i].replace(' ','.')}, inplace=True) 
    
    original_feat = list(data.columns)
    for i in range(len(original_feat)):
        data.rename(columns={original_feat[i]: original_feat[i].replace('...','.')}, inplace=True)
        data.rename(columns={original_feat[i]: original_feat[i].replace('..','.')}, inplace=True)
        
    if(test_type == 'HTTP FILE DL'):
        output_file = 'data/raw_data/d1_HttpFileDL'
    elif (test_type == 'HTTP FILE UL'):
        output_file = 'data/raw_data/d1_HttpFileUL'
    elif (test_type == 'HTTP LIVE PAGE DL'):
        output_file = 'data/raw_data/d1_HttpLivePageDL'
    elif (test_type == 'HTTP LIVE PAGE UL'):
        output_file = 'data/raw_data/d1_HttpLivePageUL'
        
        
    data = data.rename({'test.throughput.kbit.s':'transfer.datarate'}, axis='columns')
    
    data.to_csv(output_file + '_num.csv', index=False)
    data_tech.to_csv(output_file + '_cat.csv', index=False)
if (dataset == 'Nk2'):
    print('-----------------------------------------------------------------')
    print('LOADING DATASET')
    data_file = "data/DatasetA.csv"
    data = pd.read_csv(data_file, encoding = "ISO-8859-1", sep=",", decimal='.', low_memory = False)
    print('-----------------------------------------------------------------')
    
    operator  = 'Operator2'
    test      = 'HTTP Transfer UL'
    
    data = data.loc[ data['home_operator'] == operator]
    data = data.loc[ data['data_test_type'] == test]
    data = data.loc[ data['qualifier'] == 'SUCCESS' ]
    data = data.loc[~data['transfer_datarate'].isna()]
    data = data.loc[data['transfer_datarate'] > 0]
    data = data.select_dtypes(['number'])
    
    print('-----------------------------------------------------------------')
    print('Before removing duplicates\t Size:(%d, %d)'%(data.shape[0], data.shape[1]))
    data = data.T.drop_duplicates().T
    print('After removing duplicates\t Size:(%d, %d)'%(data.shape[0], data.shape[1]))
    print('-----------------------------------------------------------------')
    
    data.columns= data.columns.str.lower()
    original_feat = list(data.columns)
    for i in range(len(original_feat)):
        data.rename(columns={original_feat[i]: original_feat[i].replace('_','.')}, inplace=True)
        data.rename(columns={original_feat[i]: original_feat[i].replace(' ','.')}, inplace=True) 
    
    original_feat = list(data.columns)
    for i in range(len(original_feat)):
        data.rename(columns={original_feat[i]: original_feat[i].replace('...','.')}, inplace=True)
        data.rename(columns={original_feat[i]: original_feat[i].replace('..','.')}, inplace=True)
        
    if(test == 'Capacity DL'):
        output_file = 'data/raw_data/d2_'+ operator +'_CapacityDL.csv'
    elif (test == 'Capacity UL'):
        output_file = 'data/raw_data/d2_'+ operator +'_CapacityUL.csv'
    elif (test == 'HTTP Live'):
        output_file = 'data/raw_data/d2_'+ operator +'_HttpLive.csv'
    elif (test == 'HTTP Transfer DL'):
        output_file = 'data/raw_data/d2_'+ operator +'_HttpTransferDL.csv'
    elif (test == 'HTTP Transfer UL'):
        output_file = 'data/raw_data/d2_'+ operator +'_HttpTransferUL.csv'
        

            
    data.to_csv(output_file, index=False)