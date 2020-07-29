import pandas as pd
import os
import predict_util
import tensorflow as tf
import scipy, pickle
from models import BiRNN_new
from data_util import get_data_set, one_hot_dataset, scale, unscale, int_dataset
import random, sys, os, json
from predict_util import encode_data, get_tf_dataset, split
import bidirectional_lstm
import sys

def prepare_data(data, outpath, encoder_path):
    """Prepares the given csv file and saves it as pickle"""
    data = data.rename(index=str, columns={"Modified sequence": "Modified_sequence"})
    data['Modified_sequence'] = data['Modified_sequence'].str.replace('_','')
    data['len']=data['Modified_sequence'].str.len()

    data['CCS']=0
    data['label']=data['CCS'].values.tolist()
    split(data, outpath, 2, label_encoder_path=encoder_path)

def load_pickled_data(model_params):
    test_data = pd.read_pickle(model_params['test_file'])
    print('Using %s' % (model_params['test_file']))
    test_sequences = test_data['Modified_sequence'].values
    data = test_data
    org_columns = data.columns

    replaced_charge = False
    one_dat, lab, meta_data, test_size = encode_data(data, model_params)

    #build iterator on testdata
    dataset_test = get_tf_dataset(one_dat, lab, meta_data, data, model_params) 
    iter_test = dataset_test.make_initializable_iterator()
    next_element_test = iter_test.get_next()
    return next_element_test, meta_data, iter_test, test_size




def main():
    file_to_predict = sys.argv[1]
    data = pd.read_csv(file_to_predict)
    outfolder = './data_final/prediction'
    encoder_path = './data_final/enc.pickle'
    os.makedirs(outfolder, exist_ok=True)
    model_dir = './out/Tests_200206_ward_min2_PTtest_2/'

    encoder_path = 'data_final/enc.pickle'
    prepare_data(data, outfolder, encoder_path)
   
    with open(encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    with open(model_dir+'model_params.json') as f:
        model_params = json.load(f)
    model_params['test_file'] = os.path.join(outfolder, '2_test.pkl')
    next_element_test, meta_data, iter_test, test_size = load_pickled_data(model_params)

   
    # create graph and predict
    model = BiRNN_new
    if model_params['simple']:
        X = tf.placeholder("float", [None, model_params['num_input']])
    else:
        X = tf.placeholder("float", [None, model_params['timesteps']])
    if model_params['num_classes'] == 1:
        Y = tf.placeholder("float", [None, 1])
    else:
        Y = tf.placeholder("int64", [None, 1])
    if model_params['num_tasks'] != -1:
        T = tf.placeholder("int32", [None])
    else:
        T=None
    C = tf.placeholder("float", [None, meta_data.shape[1]])
    L = tf.placeholder("int32", [None])
    dropout = tf.placeholder("float", ())

    if model_params['num_tasks'] == -1:
        prediction, logits, weights, biases, attention, cert = model(X, C, L, model_params['num_layers'], model_params['num_hidden'],  meta_data,
                                                                    model_params['num_classes'],
                                                                    model_params['timesteps'], keep_prob=dropout,
                                                                    uncertainty=model_params['use_uncertainty'], is_train=True)
    else:
        prediction, logits, weights, biases, attention, cert = model(X, C, L, model_params['num_tasks'], model_params['num_layers'], model_params['num_hidden'], meta_data,
                                                                    model_params['num_classes'],
                                                                    model_params['timesteps'], keep_prob=dropout,
                                                                    uncertainty=model_params['use_uncertainty'], is_train=True)

    if model_params['num_classes'] == 1:    
        if model_params['num_tasks'] == -1:
            loss_op = tf.losses.mean_squared_error(predictions=prediction, labels=Y)
        else: #multitask regression.

            pp = tf.reshape(tf.stack(prediction, axis=1), [-1, model_params['num_tasks']])
            ppp = tf.reshape(tf.reduce_sum(pp * tf.one_hot(T, model_params['num_tasks']), axis=1), [-1, 1])
            loss_op = tf.losses.mean_squared_error(predictions=ppp, labels=Y)
    else:
        loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(Y,[-1]), logits=prediction)   
        loss_op = tf.reduce_mean(loss_op)
        prediction = tf.nn.softmax(prediction)
            
        
    # Initialize the variables (i.e. assign their default value)
    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    #init model
    init = [tf.global_variables_initializer(), iter_test.initializer]
    # Start training
    sess = tf.Session()
    #predictions
    sess.run(init)
    model_file = tf.train.latest_checkpoint(model_params['model_dir'])
    if model_file:
        ind1 = model_file.index('model')
        resume_itr = int(model_file[ind1+5:])
        print("Restoring model weights from " + model_file)
        saver.restore(sess, model_file)
    else:
        print('no model found!')

    label, preds, last, seq, charge, loss, att, unc, task  = bidirectional_lstm.predict(sess, X, Y, C, L, T, test_size, model_params, next_element_test, loss_op, prediction, logits, attention, meta_data, dropout, cert, dropout_rate=1.0)

    data = pd.read_csv(file_to_predict)
    data['CCS_prediction'] = preds[:, 0]
    data.to_csv(os.path.splitext(file_to_predict)[0]+'_pred.csv')



if __name__ == '__main__':
    main()




    
    