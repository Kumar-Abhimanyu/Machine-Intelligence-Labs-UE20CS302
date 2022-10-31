import numpy as np
import pandas as pd
import random

def get_entropy_of_dataset(df):
    entropy = 0

    # if df.empty:
    #     return entropy
    
    # lastColumn = df.columns[-1]
    # target = df.columns[lastColumn].unique()

    # for target_values in target:
    #     pi = df[lastColumn].value_counts()[target_values] / len(df[lastColumn])

    #     if(pi==0):
    #         continue
    #     entropy += -(pi * np.log2(pi))

    target = df[[df.columns[-1]]].values
    _, counts = np.unique(target, return_counts=True)
    total_count = np.sum(counts)

    for frequency in counts:
        temp = frequency/total_count
        if temp!=0:
            entropy -= temp*(np.log2(temp))
    return entropy

def get_avg_info_of_attribute(df,attribute):
    attr_values = df[attribute].values
    unique_attr_val = np.unique(attr_values)
    rows = df.shape[0]
    eoa = 0
    for value in unique_attr_val:
        df_slice = df[df[attribute]==value]
        target_value = df_slice[df_slice.columns[-1]].values
        _, counts = np.unique(target_value, return_counts=True)
        total_count = np.sum(counts)
        entropy = 0
        for frequency in counts:
            temp = frequency/total_count
            if temp!=0:
                entropy -= temp*(np.log2(temp))
        eoa += entropy * (np.sum(counts)/rows)
    return abs(eoa) 

def get_information_gain(df,attribute):
    info_gain = 0
    eoa = get_avg_info_of_attribute(df,attribute)
    eod = get_entropy_of_dataset(df)

    info_gain = eod - eoa
    return info_gain

def get_selected_attribute(df):
    information_gain = {}
    select_col = ''

    attr_entropy = {feature : get_avg_info_of_attribute(df,feature) for feature in df.keys()}
    #print(attr_entropy)
    for x in attr_entropy:
        #print(x)
        #print(attr_entropy[x])
        information_gain.update({x : get_information_gain(df,x)})
	 	

    selected_column = str(df.keys()[:-1][np.argmax(information_gain)])
    #print(information_gains)
    return (information_gain,selected_column)

