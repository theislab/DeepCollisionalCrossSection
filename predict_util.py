import pandas as pd
import numpy as np
import seaborn as sns
from data_util import scale, int_dataset, count_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def encode_data(data, model_params):
    test_size = len(data)
    data['lens'] = data['Modified_sequence'].str.len()
    try:
        data['task']
    except:
        data['task'] = '0'
    try:
        data['label']
    except:
        data['label'] = -1

    lab = data[model_params['lab_name']].values

    frame = [np.zeros_like(lab)]
    if model_params['num_classes'] == 1:
        min_lab, max_lab = model_params['scaling_dict']['0']
        print('minmax', min_lab, max_lab)
        lab = scale(lab, min_lab, max_lab)
        if 'Charge' in data.columns:
            frame = [data['Charge'].values]

    #scaler = StandardScaler()
    #with open('data_final/feat_enc.pickle', 'rb') as handle:
    #    scaler = pickle.load(handle)
    #frame = [scaler.transform(f.reshape(-1,1)) for f in frame]
    frame = [f.reshape(-1,1) for f in frame]
    meta_data = np.concatenate(frame, axis=1)
    if model_params['simple']:
        one_dat = count_dataset(data, model_params['timesteps'], model_params['num_input'], middle=False)
    else:
        one_dat = int_dataset(data, model_params['timesteps'], model_params['num_input'], middle=False)
    return one_dat, lab, meta_data, test_size

def get_tf_dataset(one_dat, lab, meta_data, data, model_params):
    # XXX reverse the sequences for testing of attention, could probably be done somewhere else, but here it is easy
    reverse = False
    # attention_mask = (one_dat != 22)
    # print(data['lens'].values)
    # attention_mask[:,:,data['lens'].values] = True
    # print(attention_mask)

    if reverse:
        for i, row in enumerate(one_dat):
            no_dum = row[ row != 22]
            no_dum_flip = np.flip(no_dum,0)
            one_dat[i,:len(no_dum)] = no_dum_flip[:, np.newaxis]

    if model_params['simple']:
        dx_test = tf.data.Dataset.from_tensor_slices(one_dat)
    else:
        dx_test = tf.data.Dataset.from_tensor_slices(one_dat)#.map(lambda z: tf.one_hot(z, 32))
    dy_test = tf.data.Dataset.from_tensor_slices(lab)
    dm_test = tf.data.Dataset.from_tensor_slices(meta_data)
    dl_test = tf.data.Dataset.from_tensor_slices(data['lens'].values)
    dtask_test = tf.data.Dataset.from_tensor_slices(data['task'].values)
    dataset_test = tf.data.Dataset.zip((dx_test, dm_test, dy_test, dl_test, dtask_test))
    dataset_test = dataset_test.batch(model_params['batch_size'])
    dataset_test = dataset_test.prefetch(1) 
    return dataset_test

def get_color(x, c_dict):
    cs = []
    for xx in x:
        cs.append(c_dict[xx])
    return cs


def plot_att(df, id, c_dict, remove=True, y_max=None):
    sns.set(rc={
            "axes.facecolor":"#e6e6e6",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(10.0, 6.0),
            'xtick.labelsize':12,
            'ytick.labelsize':12})
    
    seq = df['Modified_sequence'][id]
    target =  df['label'][id]
    target_pred = df['label Prediction'][id]
    att = df['attention'][id]
    
    x = np.array(list(seq)) #+ ['_' for s in range(30-len(list(seq)))]
    x_vals = np.array([xx + str(c) for c,xx in enumerate(x)])
    y = np.array(att[:len(x_vals)])
                
    if remove:
        ids = x != '_'
        x_vals = x_vals[ids]
        y = y[ids]
        x = x[ids]

    print('label', target, 'Prediction', target_pred)
    ax = plt.bar(x=range(len(y)), height=y, tick_label=x, color=get_color(x, c_dict), edgecolor='black')
    # plt.ylim(bottom=0, top=0.16)
    if not y_max is None:
        plt.ylim(bottom=0, top=y_max)
    plt.title('{} {:.2f}, {} {:.2f}'.format('label', target, 'label' + ' Prediction', target_pred))
    # plt.xticks(rotation=90)
    # plt.show()


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def split(data, name, s, label_encoder_path='data_final/enc.pickle', ids=None, calc_minval=False):
    ensure_dir(name)
    np.random.seed(s)
    with open(label_encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    data['encseq'] = data['Modified_sequence'].apply(lambda x: label_encoder.transform(list(x)))
    if calc_minval:
        data['minval'] = np.min(data['label'])
        data['maxval'] = np.max(data['label'])
    else:
        # use the CCS Values from training
        data['minval']=275.418854
        data['maxval']=1118.786133
        
    data['test']=True
    data['task'] = 0
    print('Name: ', name, 'Seed: ', s, 'Len test: ', len(data[data['test']]),'Len set test: ', len(set(data[data['test']])),'Len not test: ', len(data[~data['test']]),'Len set not test: ', len(set(data[~data['test']])))
    data[~data['test']].to_pickle(os.path.join(name, str(s) + '_train.pkl'))
    data[data['test']].to_pickle(os.path.join(name, str(s) +'_test.pkl'))
    return data
