import os
import sys
import glob

if __name__ == '__main__':
 
    print('running ccs')
    train=[

        'data_final/Tests/200206_ward_min2_PTtest/2_train.pkl'

    ]

    test=[

        'data_final/Tests/200206_ward_min2_PTtest/2_test.pkl'

    ]

    for ttrain, ttest in zip(train,test):
        mdir = '_'.join(ttrain.split('_')[:-1])
        mdir = 'out/' + '_'.join(mdir.split('/')[1:]) + '/'
        # mdir = 'out/long' + '_'.join(mdir.split('/')[1:]) + '/'
        os.system('python3 bidirectional_lstm.py {} {} {}'.format(mdir, ttrain, ttest))
