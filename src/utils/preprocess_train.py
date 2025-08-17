
import numpy as np

import pandas as pd

from torch_geometric.data import TemporalData
import torch



# first split the data for train / val /test

def train_val_test_split(df, val_ratio: float = 0.15, test_ratio: float = 0.15):
   
    val_time, test_time = np.quantile(
        df.iloc[:,2],
        [1. - val_ratio - test_ratio, 1. - test_ratio])

    val_idx = int((df.iloc[:,2] <= val_time).sum())
    test_idx = int((df.iloc[:,2] <= test_time).sum())
    
    # return traimn, val, test
    return df[:val_idx], df[val_idx:test_idx], df[test_idx:]




# func to split accirdin fixed number of event. to guarantee same train for every cic data set
def train_val_test_split_fixed(data, num_train = 200000, num_val = 70000):
   
    val_idx = int(num_train)
    test_idx = int(num_train + num_val)
    
    return data[:val_idx], data[val_idx:test_idx], data[test_idx:]


# take train and collect all rows with lable 1
# then finde what was the src_id and dest_id which corresponds to lable 1
# depending on preprocessing goal delet only labled events or all events with src_id or dst_id or both

# df_train, df_val, df_test = train_val_test_split(df=df, val_ratio = 0.15, test_ratio = 0.15)

# np.unique(df[3], return_counts=True)
def remove_out_src_dst(df_train, src_to_remove = True, dst_to_remove = False):
    
    df = df_train.loc[df_train[3] == 1].iloc[:, :4]
    src_out = df[0].unique()
    dst_out = df[1].unique()
    
    df = df_train
    if src_to_remove:
        df = df[~df.iloc[:,0].isin(src_out)]
    if dst_to_remove:
        df = df[~df.iloc[:,1].isin(dst_out)]
        
    return df


# df_train_norm = remove_out_src_dst(df_train, src_to_remove = True, dst_to_remove = False)



# return train/val/test split as temporal data for pyg tgn

def temporal_data_from_df(df, bipartite=False, attacks_names=True):
    
    src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
    dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
    
    if bipartite:
        dst += int(src.max()) + 1
    
    t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
    y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
    
    # check if we have attack names in data
    y_label=None
    if attacks_names:
        y_label = df.iloc[:, 4].values
        df = df.drop(columns=4)
    
    msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)
    
    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
    return data#, y_label







def get_split_clean_train(df, val_ratio = 0.15, test_ratio = 0.15,
                          src_to_remove = True, dst_to_remove = False,
                          bipartite=False, attacks_names=False
                          ):
    
    df_train, df_val, df_test = train_val_test_split(df=df, val_ratio=0,
                                                     test_ratio=test_ratio)
    
    
    df_train_norm = remove_out_src_dst(df_train, src_to_remove = src_to_remove, 
                                       dst_to_remove = dst_to_remove)
    
    
    df_new = pd.concat([df_train_norm, df_val, df_test], axis=0)
    
    new_test_ratio = df_test.shape[0]/df_new.shape[0]
    print('adjusted test ratio after obtaining new train', new_test_ratio)    
    data_new = temporal_data_from_df(df_new, bipartite=bipartite, attacks_names=attacks_names)
    
    train_data, val_data, test_data = data_new.train_val_test_split(
        val_ratio=val_ratio, test_ratio=new_test_ratio)
        
    return data_new, train_data, val_data, test_data

# the same as above but only return hovle temporal data and ne test split(where class1 established)
def get_clean_train(df, val_ratio = 0.15, test_ratio = 0.15,
                          src_to_remove = True, dst_to_remove = False,
                          bipartite=False, attacks_names=False
                          ):
    
    df_train, df_val, df_test = train_val_test_split(df=df, val_ratio=0,
                                                     test_ratio=test_ratio)
    
    
    df_train_norm = remove_out_src_dst(df_train, src_to_remove = src_to_remove, 
                                       dst_to_remove = dst_to_remove)
    
    
    df_new = pd.concat([df_train_norm, df_val, df_test], axis=0)
    
    new_test_ratio = df_test.shape[0]/df_new.shape[0]
    print('adjusted test ratio after obtaining new train', new_test_ratio)    
    data_new = temporal_data_from_df(df_new, bipartite=bipartite, attacks_names=attacks_names)
    
    # train_data, val_data, test_data = data_new.train_val_test_split(
    #     val_ratio=val_ratio, test_ratio=new_test_ratio)
        
    return data_new, new_test_ratio


