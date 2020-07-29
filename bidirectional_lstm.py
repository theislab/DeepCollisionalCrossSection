from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, os, json
from models import BiRNN_new
#from models import BiRNN_attention, BiRNN_attention_multitask, conv_net, mlp, logreg, Transformer
from data_util import get_data_set, unscale
from sklearn.metrics import accuracy_score
from data_util import DUMMY, Delta_t95
import time

np.random.seed(13)
tf.set_random_seed(13)
random.seed(22)
classification = False

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def predict(sess, X, Y, C, L, T, test_size, model_params, next_element_test, loss_op, prediction, logits, attention, meta_data, dropout, cert, dropout_rate = 1.0):
        # Calculate accuracy for 128 mnist test images
        its = test_size//model_params['batch_size']
        err = 0
        preds = np.zeros([test_size, model_params['num_classes']], dtype=np.float32)
        label = np.zeros([test_size, 1], dtype=np.float32)
        if model_params['simple']:
            seq = np.zeros([test_size, model_params['num_input']], dtype=np.float32)
        else:
            #seq = np.ones([test_size, model_params['timesteps'], model_params['num_input']], dtype=np.float32)*DUMMY
            seq = np.ones([test_size, model_params['timesteps']], dtype=np.float32)*DUMMY
        charge = np.zeros([test_size, meta_data.shape[1]], dtype=np.float32)
        llast = np.zeros([test_size, model_params['num_hidden']], dtype=np.float32)
        if attention is None:
            att = np.zeros([test_size, model_params['timesteps']], dtype=np.float32)
        else:
            att = np.zeros([test_size, np.max(attention.shape)], dtype=np.float32)
        uncert = np.zeros([test_size], dtype=np.float32)
        tasks = np.zeros([test_size], dtype=np.float32)

        if test_size % model_params['batch_size'] != 0:
            its += 1

        for i in range(its):
            batch_x, batch_c, batch_y, batch_l, batch_t = sess.run(next_element_test)
            batch_x = np.squeeze(batch_x)
            batch_y = np.reshape(batch_y, [batch_y.shape[0], 1])
            batch_c = np.reshape(batch_c, [batch_y.shape[0], meta_data.shape[1]])

            if cert != None:
                loss, p, llast_, a, c = sess.run([loss_op, prediction, logits, attention, cert], feed_dict={X: batch_x, Y: batch_y, C: batch_c, dropout:dropout_rate})
            elif attention != None:

                feed_dict = {X: batch_x, Y: batch_y, C: batch_c, L: batch_l, dropout: dropout_rate}
                if model_params['num_tasks'] != -1:
                    feed_dict[T] = batch_t

                loss, p, llast_, a = sess.run([loss_op, prediction, logits, attention], feed_dict=feed_dict)
                #c = np.zeros([p.shape[0],1])
            else:
                loss, p, llast_ = sess.run([loss_op, prediction, logits], feed_dict={X: batch_x, Y: batch_y, 
                                                                                     C: batch_c, dropout:1.0,
                                                                                     L: batch_l})
                #c = np.zeros([p.shape[0],1])
                a = np.zeros([p.shape[0], model_params['timesteps'], 1])

            shape = preds[i*model_params['batch_size']:(i+1)*model_params['batch_size']].shape[0]
                
            if model_params['num_tasks'] != -1:
                p = np.hstack(p)
                p = p[range(p.shape[0]), batch_t].reshape([-1,1])

            preds[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = p[:shape]
            label[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = batch_y[:shape]
            charge[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = batch_c[:shape]
            seq[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = batch_x[:shape]
            tasks[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = batch_t[:shape]
            llast[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = llast_[:shape]
            # print('a', a.shape, shape)
            att[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = a[:shape, :, 0]
            # print('att', att.shape)
            #uncert[i*model_params['batch_size']:(i+1)*model_params['batch_size']] = c[:shape, 0]
            
            err += loss

        if model_params['num_classes'] == 1:
            #label = unscale(label, model_params['min_RT'], model_params['max_RT'] ) # * 100#np.exp(label)
            #preds = unscale(preds, model_params['min_RT'], model_params['max_RT'] )# * 100 #np.exp(preds)

            log = False
            if not log:
                for tt in range(np.maximum(1, model_params['num_tasks'])):
                    preds[tasks == tt] = unscale(preds[tasks == tt], model_params['scaling_dict'][str(tt)][0],
                                            model_params['scaling_dict'][str(tt)][1])
                    label[tasks == tt] = unscale(label[tasks == tt], model_params['scaling_dict'][str(tt)][0],
                                            model_params['scaling_dict'][str(tt)][1])
            else:
                preds = np.exp(preds)
                label = np.exp(label)

        #print("Test loss", err / (its), np.mean( (preds-label)**2 ))

        return label, preds, llast, seq, charge, err / (its), att, uncert, tasks

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    m_dir = sys.argv[1]
    train_data = sys.argv[2]
    test_data = sys.argv[3]

    SAVE_INTERVAL = 10000
    EVAL_interval = 5000

    # Training Parameters
    display_step = 200

    lab_name = 'label'#''label'
    model_dir = m_dir #'./out_new/'+lab_name+'_attbilstm_multitask_ccs_mod/'
    model_dir_pretrain = None#'/media/niklas/Data_3/MannProteomics_jean/MannProteomics/out/_old_Tests_data_detectabiliy_sml_2'#'/home/jean/Prog/hiwi/MannProteomics/out/Tests_RetentionCorrected2mio_2'#''./out/'+lab_name+'_attbilstm_multitask_mod/'

    ensure_dir(model_dir)

    model_params = {}

    # Network Parameters
    model_params['lab_name'] = 'label'#PeptideFound'#lab_name
    model_params['fname'] = 'cache/one_dat_cache_full_'+lab_name+'.npy'
    model_params['num_input'] = 32 # MNIST data input (img shape: 28*28)
    model_params['timesteps'] = 66 # timesteps
    model_params['num_hidden'] = 128
    model_params['num_layers'] = 2

    model_params['num_classes'] = 1 # MNIST total classes (0-9 digits)
    model_params['dropout_keep_prob'] = 0.9#0.75#1.0
    model_params['use_uncertainty'] = False
    model_params['use_attention'] = True
    model_params['simple'] = False
    model_params['num_tasks'] = -1#10
    model_params['batch_size'] = 64
    model_params['model_dir'] = model_dir
    model_params['model_dir_pretrain'] = model_dir_pretrain
    model_params['lr_base'] = 0.001
    model_params['training_steps'] = 55000#56000
    model_params['reduce_lr_step'] = 50000#50000#35000

    if model_params['num_tasks'] == -1:
        model = BiRNN_new
    else:
        model = BiRNN_attention_multitask
    if model_params['simple']:
        model = mlp

   
 

    ensure_dir('./cache/')

    figure_dir = model_params['model_dir'] + 'figures/'
    ensure_dir(figure_dir)

    model_params['train_file'] = train_data#'data/20180904_CCS_library_raw_removedduplicates_train.pkl'
    model_params['test_file'] = test_data#'data/mod_test_2.pkl'

    if classification:
        model_params['num_classes'] = 5 # MNIST total classes (0-9 digits)


    model_params['reduce_train'] = 0.5

    dataset, dataset_test, train_size, test_size, meta_data  = get_data_set(model_params)
    print(model_params)
    with open(model_params['model_dir'] +'model_params.json', 'w') as f:
        json.dump(model_params, f)
    #construct iterators
    dataset = dataset.shuffle(100000, reshuffle_each_iteration=True)
    dataset = dataset.repeat(None)
    #dataset = dataset.apply(tf.contrib.data.unbatch())

    dataset = dataset.batch(model_params['batch_size'])
    dataset = dataset.prefetch(1) 
    dataset_test = dataset_test.batch(model_params['batch_size'])
    dataset_test = dataset_test.prefetch(1) 
    iter = dataset.make_initializable_iterator()
    next_element = iter.get_next()
    iter_test = dataset_test.make_initializable_iterator()
    next_element_test = iter_test.get_next()

    #build graph
    if not model_params['simple']:
        #X = tf.placeholder("float", [None, model_params['timesteps'], model_params['num_input']])
        X = tf.placeholder("float", [None, model_params['timesteps']])
    else:
        X = tf.placeholder("float", [None, model_params['num_input']])

    if model_params['num_classes'] == 1:
        Y = tf.placeholder("float", [None, 1])
    else:
        Y = tf.placeholder("int64", [None, 1])

    if model_params['num_tasks'] != -1:
        T = tf.placeholder("int32", [None])

    C = tf.placeholder("float", [None, meta_data.shape[1]])
    L = tf.placeholder("int32", [None])

    dropout = tf.placeholder("float", ())
    if model_params['num_tasks'] == -1:
        prediction, logits, weights, biases, attention, cert = model(X, C, L, model_params['num_layers'], model_params['num_hidden'],  meta_data,
                                                                     model_params['num_classes'],
                                                                     model_params['timesteps'], keep_prob=dropout,
                                                                     uncertainty=model_params['use_uncertainty'])
    else:
        prediction, logits, weights, biases, attention, cert = model(X, C, L, model_params['num_tasks'], model_params['num_layers'], model_params['num_hidden'], meta_data,
                                                                     model_params['num_classes'],
                                                                     model_params['timesteps'], keep_prob=dropout,
                                                                     uncertainty=model_params['use_uncertainty'])


    if model_params['num_classes'] == 1:
        if not model_params['use_uncertainty']: #standard regression

            if model_params['num_tasks'] == -1:
                loss_op = tf.losses.mean_squared_error(predictions=prediction, labels=Y)
            else: #multitask regression.

                pp = tf.reshape(tf.stack(prediction, axis=1), [-1, model_params['num_tasks']])
                ppp = tf.reshape(tf.reduce_sum(pp * tf.one_hot(T, model_params['num_tasks']), axis=1), [-1, 1])
                loss_op = tf.losses.mean_squared_error(predictions=ppp, labels=Y)

        else:
            loss_op = tf.pow(prediction - Y, 2)/2 * tf.exp(-cert) + 1/2*cert
            loss_op = tf.reduce_mean(loss_op)
    else:
        loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(Y,[-1]), logits=prediction)   
        loss_op = tf.reduce_mean(loss_op)

    
    learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables (i.e. assign their default value)
    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=1)

    pearson_ph = tf.placeholder(tf.float32, shape=())
    ce_ph = tf.placeholder(tf.float32, shape=())
    rel_std_ph = tf.placeholder(tf.float32, shape=())
    deltat_ph = tf.placeholder(tf.float32, shape=())

    if model_params['num_classes'] == 1:
        tf.summary.scalar('pearson', pearson_ph)
        tf.summary.scalar('relative error std', rel_std_ph)
        tf.summary.scalar('r2', deltat_ph)
    else:
       tf.summary.scalar('accuracy', pearson_ph)
    tf.summary.scalar('cross_entropy', ce_ph)

    #init model
    init = [tf.global_variables_initializer(), iter.initializer, iter_test.initializer]
    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(model_params['model_dir'] + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(model_params['model_dir'] + 'test')



    sess.run(init)

    #load pretrained
    if model_params['model_dir_pretrain'] != None:
        model_file = tf.train.latest_checkpoint(model_params['model_dir_pretrain'])
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            saver.restore(sess, model_file)
            print("restored model weights from " + model_file)
        else:
            print('No model file found in: ', model_params['model_dir_pretrain'])
            sys.exit()
    else:
        resume_itr = 0

    #training
    start = time.time()
    for step in range(resume_itr, model_params['training_steps']+1):
        if step <= model_params['reduce_lr_step']:
            lr = model_params['lr_base']
        else:
            lr = model_params['lr_base'] / 10.

        # Run optimization op (backprop)

        batch_x, batch_c, batch_y, batch_l, batch_t = sess.run(next_element)
        batch_x = np.squeeze(batch_x)
        batch_y = np.reshape(batch_y, [model_params['batch_size'], 1])
        batch_c = np.reshape(batch_c, [model_params['batch_size'], meta_data.shape[1]])

        feed_dict={X: batch_x, Y: batch_y, C: batch_c, L: batch_l,dropout:model_params['dropout_keep_prob'], learning_rate : lr}
        if model_params['num_tasks'] != -1:
            feed_dict[T] = batch_t

        sess.run(train_op, feed_dict=feed_dict)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            fetch = [loss_op, prediction]
            feed_dict = {X: batch_x, Y: batch_y, C: batch_c, L: batch_l, dropout:1.0}
            if model_params['num_tasks'] != -1:
                feed_dict[T] = batch_t

            loss, pred = sess.run(fetch, feed_dict=feed_dict)
            if model_params['num_tasks'] != -1:
                pred = np.hstack(pred)
                pred = pred[range(pred.shape[0]), batch_t].reshape([-1,1])
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(loss) + " time elapsed: {:.4f}s".format(time.time() - start))
            start = time.time()
            pred = np.squeeze(pred)
            batch_y = np.squeeze(batch_y)

            if model_params['num_classes'] == 1:
                pearson = scipy.stats.pearsonr(pred, batch_y)[0]
                from sklearn.metrics import r2_score
                r2=r2_score(batch_y,pred)
                #deltat = 0#Delta_t95(batch_y, pred)[0]
                rel = (batch_y - pred)/batch_y * 100# - 100
                feed = {pearson_ph : pearson,
                        ce_ph : loss,
                        rel_std_ph : np.abs(rel).mean(),
                        deltat_ph : r2}
            else:
                acc = accuracy_score(np.argmax(pred,axis=1), batch_y)
                feed = {pearson_ph : acc,
                        ce_ph : loss,
                        }

            if step != 1:
                summary = sess.run(merged, feed_dict=feed)
                train_writer.add_summary(summary, step)

        if step % EVAL_interval == 0:
            if model_params['num_tasks'] == -1:
                T = None
            label, preds, last, seq, charge, loss, att, c , tasks = predict(sess, X, Y, C, L, T, test_size, model_params, next_element_test, loss_op, prediction, logits, attention, meta_data, dropout, cert)
            preds = np.squeeze(preds)
            label = np.squeeze(label)
            if model_params['num_classes'] == 1:
                # XXX somehow test error is smaller, but test pearson also, train delta_t95 is also bugged, does not make sense!
                pearson = scipy.stats.pearsonr(preds, label)[0]
                rel = (label - preds)/label*100#) * 100 - 100
                #deltat = 0#Delta_t95(label, preds)[0]
                from sklearn.metrics import r2_score
                r2=r2_score(label,preds)
                feed={pearson_ph : pearson,
                           ce_ph : loss,
                      rel_std_ph : np.abs(rel).mean(),
                      deltat_ph : r2}
            else:
                acc = accuracy_score(np.argmax(preds,axis=1), label)
                feed = {pearson_ph : acc,
                        ce_ph : loss,
                        }

            summary = sess.run(merged, feed_dict=feed)
            sess.run(iter_test.initializer)
            test_writer.add_summary(summary, step)

        if (step!=0) and step % SAVE_INTERVAL == 0:
            saver.save(sess, model_params['model_dir'] + '/' + 'model' + str(step))
            print('model saved')

        #reporter(model_params['timesteps']_total=step, loss=loss) # report metrics
    saver.save(sess, model_params['model_dir'] + '/' + 'model' + str(step))
    print("Model_saved, Optimization Finished!")
