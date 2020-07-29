import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import random, sys, os
import tensorflow as tf
import pickle


np.random.seed(13)
random.seed(22)
tf.set_random_seed(13)
classification = False
DUMMY = 22


def RMSE(act, pred):
    '''
    accept two numpy arrays
    '''
    return np.sqrt(np.mean(np.square(act - pred)))

from scipy.stats import pearsonr
def Pearson(act, pred):
    return pearsonr(act, pred)[0]

from scipy.stats import spearmanr
def Spearman(act, pred):
    '''
    Note: there is no need to use spearman correlation for now
    '''
    return spearmanr(act, pred)[0]

def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]

def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))


def one_hot_dataset(dat, label_encoder, onehot_encoder, timesteps, num_input, middle = True):
    oh_dat = np.zeros([len(dat), timesteps, num_input])
    for c, el in enumerate(dat):
        ie = label_encoder.transform(el)
        #print(ie)
        ie = ie.reshape(len(ie), 1)
        oe = np.array(onehot_encoder.transform(ie))
        #oh_dat[c, 0:oe.shape[0], :] = oe
        if middle:
            oh_dat[c, ((60-oe.shape[0])//2): ((60-oe.shape[0])//2)+oe.shape[0], :] = oe
        else:
            oh_dat[c, 0:oe.shape[0], :] = oe

    return oh_dat


def int_dataset(dat, timesteps, num_input, middle = True):
    oh_dat = (np.ones([len(dat), timesteps, 1], dtype=np.int32)*DUMMY).astype(np.int32)

    cnt = 0
    for c, row in dat.iterrows():
        ie = np.array(row['encseq'])
        oe = ie.reshape(len(ie), 1)
        if middle:
            oh_dat[cnt, ((60-oe.shape[0])//2): ((60-oe.shape[0])//2)+oe.shape[0], :] = oe
        else:
            oh_dat[cnt, 0:oe.shape[0], :] = oe
        cnt += 1

    return oh_dat

def count_dataset(dat, timesteps, num_input, middle=False):
    ds = np.zeros([len(dat), num_input], dtype=np.int32)
    cnt = 0
    for c, row in dat.iterrows():
        seq = row['encseq']
        for v in seq:
            ds[cnt, v] += 1
        ds[cnt, -1] = np.sum(ds[cnt])
        cnt += 1
        if middle:
            raise NotImplementedError
    return ds

def scale(lab, min_lab, max_lab):
    return (lab - min_lab) / (max_lab - min_lab)

def unscale(lab, min_lab, max_lab):
    return (max_lab - min_lab) * lab + min_lab

#convert to int encoded dataset, tf one hot for training
def get_data_set(model_params, return_array=False, middle=False):
    do_one_hot = False
    reverse = False
    intrinsic_test = model_params['train_file'] == model_params['test_file']

    sname = 'Modified_sequence'
    data=pd.read_pickle(model_params['train_file'])
    data = data.sample(frac=1).reset_index(drop=True)

    if not intrinsic_test:
        data_test = pd.read_pickle(model_params['test_file'])
        data = data.sample(frac=1).reset_index(drop=True)
        test_from = len(data)
        data = pd.concat([data, data_test], ignore_index=True, sort=False)
        
    #lab_name = 'Retention_time'
    if model_params['num_classes'] == 1:
        #lab = np.log(data[model_params['lab_name']].values)
        lab = data[model_params['lab_name']].values
        min_lab = data['minval']
        max_lab = data['maxval']

        log = False

        if not log:
            lab = scale(lab, min_lab, max_lab).values
        else:
            lab = np.log(lab)

        scaling_dict = {}
        for i in range(np.maximum(model_params['num_tasks'], 1)):
            a = data[data['task'] == i]['minval'].values[0]
            b = data[data['task'] == i]['maxval'].values[0]
            scaling_dict[str(i)] = (float(a),float(b))
        
        model_params['scaling_dict'] = scaling_dict

    else:
        lab = data[model_params['lab_name']].values
    
    dat = data[sname].values
    if model_params['num_classes'] == 1 and 'Charge' in data.columns:
        frame = [data['Charge'].values]
    else:
        frame = [np.zeros_like(lab)]
    
    if intrinsic_test:
        #sort then shuffle sinse set since ordering in set is non deterministic
        seqs = list(sorted(set(dat)))
        np.random.shuffle(seqs)
        idx = int(0.9*len(seqs))

        #split training test
        training_seqs, test_seqs = seqs[:idx], seqs[idx:]
        print('unique sequences training', len(training_seqs),'test',len(test_seqs))

        #store into dataframe
        dd = pd.DataFrame({'seqs' : dat})
        training_idx = dd[dd['seqs'].isin(training_seqs)].index.astype(int)
        test_idx = dd[dd['seqs'].isin(test_seqs)].index.astype(int)
    else:
        training_idx = data[:test_from].index.astype(int)
        test_idx = data[test_from:].index.astype(int)
    
    # print('first training ids:', training_idx[:20])

    if model_params['reduce_train'] != 1.0:
        training_idx = training_idx[:int(len(training_idx)*model_params['reduce_train'])]

    #scale features if necessary
    if len(frame) != 0:
        #scaler = StandardScaler()
        #with open('data/feat_enc.pickle', 'rb') as handle:
        #    scaler = pickle.load(handle)
    
        #frame = [scaler.transform(f.reshape(-1,1)) for f in frame]
        frame = [f.reshape(-1,1) for f in frame]

        #with open('data/feat_enc.pickle', 'wb') as handle:
        #    pickle.dump(scaler, handle)  
    
        
    meta_data = np.concatenate(frame, axis=1)
    lab = np.reshape(lab, [lab.shape[0], 1])


    data['lens'] = data['Modified_sequence'].str.len()
    dtest = data.iloc[test_idx]
    dtrain = data.iloc[training_idx]



    #simple just uses basic features and len of sequence
    if model_params['simple']:
        # some unnecessary code here
        count_dat = count_dataset(data, model_params['timesteps'], model_params['num_input'], middle=False)
        dx_train = tf.data.Dataset.from_tensor_slices(count_dat[training_idx,:])
        dx_test = tf.data.Dataset.from_tensor_slices(count_dat[test_idx,:])

        dy_train = tf.data.Dataset.from_tensor_slices(lab[training_idx])
        dm_train = tf.data.Dataset.from_tensor_slices((meta_data[training_idx, :]))
        dl_train = tf.data.Dataset.from_tensor_slices(dtrain['lens'].values)
        dtask_train = tf.data.Dataset.from_tensor_slices(dtrain['task'].values)
        dataset_train = tf.data.Dataset.zip((dx_train, dm_train, dy_train, dl_train, dtask_train))

        dy_test = tf.data.Dataset.from_tensor_slices(lab[test_idx])
        dm_test = tf.data.Dataset.from_tensor_slices(meta_data[test_idx, :])
        dl_test = tf.data.Dataset.from_tensor_slices(dtest['lens'].values)
        dtask_test = tf.data.Dataset.from_tensor_slices(dtest['task'].values)
        dataset_test = tf.data.Dataset.zip((dx_test, dm_test, dy_test, dl_test, dtask_test))
    else:
        one_dat = int_dataset(data, model_params['timesteps'], model_params['num_input'], middle=False)
        if not do_one_hot:

            if reverse:
                for i, row in enumerate(one_dat):
                    no_dum = row[ row != DUMMY]
                    no_dum_flip = np.flip(no_dum,0)
                    one_dat[i,:len(no_dum)] = no_dum_flip[:, np.newaxis]

            #dx_train = tf.data.Dataset.from_generator(lambda: dataset_list, tf.int32), tf.TensorShape([None, 32, 2])).map(lambda z: tf.one_hot(z, 32))
            #dx_train = tf.data.Dataset.from_generator(lambda: b, tf.int32).map(lambda z: tf.one_hot(z, 32))

            dx_train = tf.data.Dataset.from_tensor_slices(one_dat[training_idx,:])#.map(lambda z: tf.one_hot(z, 32))

            dy_train = tf.data.Dataset.from_tensor_slices(lab[training_idx])
            dm_train = tf.data.Dataset.from_tensor_slices((meta_data[training_idx, :]))
            dl_train = tf.data.Dataset.from_tensor_slices(dtrain['lens'].values)
            dtask_train = tf.data.Dataset.from_tensor_slices(dtrain['task'].values)

            dataset_train = tf.data.Dataset.zip((dx_train, dm_train, dy_train, dl_train, dtask_train))

            dx_test = tf.data.Dataset.from_tensor_slices(one_dat[test_idx,:])#.map(lambda z: tf.one_hot(z, 32))
            #dx_test = tf.data.Dataset.from_generator(lambda: a, tf.int32, output_shapes=[None]).map(lambda z: tf.one_hot(z, 32))


            dy_test = tf.data.Dataset.from_tensor_slices(lab[test_idx])
            dm_test = tf.data.Dataset.from_tensor_slices(meta_data[test_idx, :])
            dl_test = tf.data.Dataset.from_tensor_slices(dtest['lens'].values)
            dtask_test = tf.data.Dataset.from_tensor_slices(dtest['task'].values)

            dataset_test = tf.data.Dataset.zip((dx_test, dm_test, dy_test, dl_test, dtask_test))
        else:
            dataset_train = tf.data.Dataset.from_tensor_slices((one_dat[training_idx,:],meta_data[training_idx, :],lab[training_idx]))
            dataset_test = tf.data.Dataset.from_tensor_slices((one_dat[test_idx,:],meta_data[test_idx, :],lab[test_idx]))



    dataset, dataset_test, train_size, test_size = dataset_train, dataset_test, dtrain.shape[0], dtest.shape[0]
    print('done generating data trainsize:', train_size,'testsize' ,test_size)

    return dataset, dataset_test, train_size, test_size, meta_data,
