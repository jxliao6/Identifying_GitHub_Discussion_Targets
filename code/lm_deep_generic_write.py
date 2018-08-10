import argparse, hashlib, csv, random,pandas
import numpy as np

from format_lm_input_write import create_committer_data_dict, create_comments_dict, \
                            train_test_split_idxs, get_index_tokenizer, build_data, \
                            trim_pad_sequences, flatten_comments_dict, mention_preprocess,\
                            remove_short_comments, Using_one_project, randomizing_data,\
                            length_sequences
from format_lm_input_write import COMMIT_METRICS

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score


MODEL_TYPES = ['classifier', 'commits']
DATA_TYPES = ['speaking', 'spoken_to']

class ComputeAUC(Callback):

    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        print('')
        print('')
        print('Computing AUC for epoch %d' % epoch)
        y_score = self.model.predict(self.validation_data[0], batch_size=512)
        auc = float(roc_auc_score(self.validation_data[1], y_score)) * 100
        self.history.append(auc)
        print('')
        print('AUC: %.5f%%' % auc)
        with open("result", "a") as myfile:
            myfile.write(' %.2f%%' % auc)


class ComputeConfusionMatrix(Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('')
        print('Computing confusion matrix for epoch %d' % epoch)
        y_preds_cls = self.model.predict_classes(self.validation_data[0], batch_size=512)
        conf_mat = confusion_matrix(self.validation_data[1], y_preds_cls)
        print('')
        print('Confusion matrix:')
        conf_mat = confusion_matrix(self.validation_data[1], y_preds_cls)
        print(conf_mat)
        print('')

def do_classifier(args, X_train, y_train, X_test, y_test, epochs):
    embedding_length = 64

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)

    # Need to trim input data to some max length of tokens
    model = Sequential()
    model.add(Embedding(input_dim=len(tkr.word_index) + 2,
                        output_dim=embedding_length,
                        input_length=args.max_seq_len,
                        mask_zero=False,
                        trainable=True))
    model.add(Conv1D(filters=embedding_length,
                            padding='same',
                            activation='relu',
                            kernel_size=6))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(30, activation='relu', dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')

    with open("result", "a") as myfile:
        myfile.write('AUC after each epoch:')

    history = model.fit(X_train_np, y_train_np,
                        validation_data=(X_test_np, y_test_np),
                        epochs=epochs,
                        batch_size=512,
                        callbacks=[ComputeAUC(), ComputeConfusionMatrix()]
                        )

    y_score = model.predict(X_test_np, batch_size=512)
    auc = float(roc_auc_score(y_test_np, y_score)) * 100
    print('')
    print('Final AUC: %.5f%%' % auc)

    y_preds_cls = model.predict_classes(X_test_np, batch_size=512)
    conf_mat = confusion_matrix(y_test_np, y_preds_cls)
    print('')
    print('Final Confusion matrix:')
    conf_mat = confusion_matrix(y_test_np, y_preds_cls)
    print(conf_mat)
    with open("result", "a") as myfile:
        myfile.write('\nFinal Confusion matrix: \n')
        myfile.write(str(conf_mat[0][0])+' '+str(conf_mat[0][1])+'\n')
        myfile.write(str(conf_mat[1][0])+' '+str(conf_mat[1][1])+'\n')
        myfile.write('\n')

    # bad_guess_idxs = []
    # for idx in range(len(y_test)):
    #     if y_test[idx] != y_preds_cls[:,0][idx]:
    #         bad_guess_idxs.append(idx)

    # bad_guess_comments = [comments_dict_flat_test[i] for i in bad_guess_idxs]

    # with open('./data/bad_guess_comments.csv', 'w') as outf:
    #     w = csv.DictWriter(outf, fieldnames=bad_guess_comments[0].keys())
    #     w.writeheader()
    #     for comment in bad_guess_comments:
    #         w.writerow(comment)


def do_regression(args, X_train, y_train, X_test, y_test):
    embedding_length = 64

    # Need to trim input data to some max length of tokens
    model = Sequential()
    model.add(Embedding(input_dim=len(tkr.word_index) + 1,
                        output_dim=embedding_length,
                        input_length=args.max_seq_len,
                        mask_zero=False,
                        trainable=True))
    # model.add(Convolution1D(nb_filter=embedding_length,
    #                         filter_length=6,
    #                         border_mode='same',
    #                         activation='relu',
    #                         init='normal'))
    # model.add(MaxPooling1D(pool_length=5))
    # model.add(LSTM(10, activation='relu', dropout_W=0.2, init='normal'))
    model.add(Flatten())
    model.add(Dense(10, init='normal'))
    model.add(Dense(10, init='normal'))
    model.add(Dense(10, init='normal'))
    model.add(Dense(10, init='normal'))
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error',
                  # metrics=[r2_score],
                  optimizer='adam')

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        nb_epoch=args.epochs,
                        batch_size=512)


def do_model(args, X_train, y_train, X_test, y_test, epochs):
    if args.model_type == 'classifier':
        do_classifier(args, X_train, y_train, X_test, y_test, epochs)
    else:
        do_regression(args, X_train, y_train, X_test, y_test, epochs)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Formats data for LM input')
    argparser.add_argument('base_comment_dir',
                           action='store',
                           help='Base directory of issue comments',
                           default='./data_disjoint_random_10_48_wclosed')
    argparser.add_argument('base_commit_dir',
                           action='store',
                           help='Base directory of commits',
                           default='./data_disjoint_random_10_48_commits')
    argparser.add_argument('comments_file',
                           action='store',
                           help='Comments file name',
                           default='issue_comments_REPLACECODE.csv')
    argparser.add_argument('model_type',
                           action='store',
                           help='Model type',
                           choices=MODEL_TYPES,
                           default='classifier')
    argparser.add_argument('data_type',
                           action='store',
                           help='Data type',
                           choices=DATA_TYPES,
                           default='speaking')
    argparser.add_argument('--top_perc',
                           action='store',
                           help='Top percent of the response metric to be the positive class',
                           type=int)
    argparser.add_argument('--metric',
                           action='store',
                           help='Which metric do we want for the top percent?',
                           choices=COMMIT_METRICS,
                           type=str)
    argparser.add_argument('--max_seq_len',
                           action='store',
                           help='Maximum sequence (post) length to deal with. Larger will be trimmed, shorter will be padded.',
                           type=int,
                           default=32)
    argparser.add_argument('--epochs',
                           action='store',
                           help='Number of epochs',
                           type=int,
                           default=10)
    argparser.add_argument('--noise_len',
                           action='store',
                           help='Remove post that is shorter than the given length.',
                           type=int,
                           default=0)
    argparser.add_argument('--test_project',
                           action='store',
                           help='Test on one project and train on all other projects.',
                           type=int,
                           default=-1)
    argparser.add_argument('--one_project',
                           action='store',
                           help='Only use data from one project.',
                           type=int,
                           default=-1)
    argparser.add_argument('--baseline',
                           action='store_true',
                           help='Randomize data')
    argparser.add_argument('--shuffle',
                           action='store_true',
                           help='Shuffle data')
    argparser.add_argument('--equal2modelsize',
                            action='store_true',
                            help='sample speaking datasize as spoken per project')
    argparser.add_argument('--equal2fullmodelsize',
                            action='store_true',
                            help='sample speaking datasize as spoken for all projects')
    argparser.add_argument('--datasize_file',
                            action='store',
                            help='data size of sampling speaking datasize as spoken per project')
    argparser.add_argument('--tokenlength_only',
                            action='store_true',
                            help = 'test on the effect of token length')                      

    args = argparser.parse_args()

    with open("result", "a") as myfile:
        myfile.write('Model: ')
        myfile.write(args.data_type)
        myfile.write('\n')
        myfile.write('Min Sequence Length: ')
        myfile.write(str(args.noise_len))
        myfile.write('\n')
        myfile.write('Max Sequence Length: ')
        myfile.write(str(args.max_seq_len))
        myfile.write('\n')

    committer_dict = create_committer_data_dict(args.base_commit_dir, args.top_perc, args.metric)
    h = hashlib.sha1()
    h.update(('format_lm_input_write.py' + args.base_comment_dir + args.base_commit_dir + args.comments_file).encode('utf-8'))
    comments_dict = create_comments_dict(args.base_comment_dir, committer_dict, args.comments_file, args.data_type, h)

    if args.one_project >= 0:
        print('Using one project')
        if args.test_project >= 0:
            raise
        comments_dict,projectname = Using_one_project(args.one_project, comments_dict)

    print('Removing noise')
    comments_dict = remove_short_comments(comments_dict, args.noise_len)

    print('Splitting idxs')
    if args.test_project == -1:
        train_idxs, test_idxs = train_test_split_idxs(comments_dict, args.test_project)
    else:
        train_idxs, test_idxs,projectname = train_test_split_idxs(comments_dict, args.test_project)
    tkr = get_index_tokenizer(comments_dict, train_idxs)
    
    print('Building data')
    X, y, ids = build_data(comments_dict, committer_dict, tkr, args.model_type, args.data_type)

    if args.data_type == 'spoken_to':
        print('Trimming @mention')
        X = mention_preprocess(X, ids, args.max_seq_len)

    print('Converting texts to sequences')
    X = tkr.texts_to_sequences(X)
    # tkr.word_index starts indexing at 1 not 0, so the pad idx is len + 1
    
    if args.tokenlength_only:
        print('Converting all words into same token')
        X=length_sequences(X)
        
        with open('result','a') as myfile:
            myfile.write('sequence length test\n')

    print('Trimming and padding to max_seq_len %d' % args.max_seq_len)
    X = trim_pad_sequences(X, args.max_seq_len, len(tkr.word_index) + 1)

    print('Flattening comments dict')
    comments_dict_flat = flatten_comments_dict(comments_dict)

    with open("result", "a") as myfile:
        myfile.write('Train size: ')
        myfile.write(str(len(train_idxs)))
        myfile.write('\n')
        myfile.write('Test size: ')
        myfile.write(str(len(test_idxs)))
        myfile.write('\n')

    if args.baseline:
        print("Randomizing data")
        X2 = []
        for x in X:
            x2 = []
            for xx in x:
                x2.append(random.randint(1, len(tkr.word_index)))
            X2.append(x2)
        X = X2

    if args.shuffle:
        print("shuffling data")
        for ind, _ in enumerate(X):
            random.shuffle(X[ind])

    if args.equal2modelsize:
        datasize = pandas.read_csv(args.datasize_file)
        print(projectname)
        trainsize = int(datasize[datasize['project']==projectname]['trainsize'])
        testsize = int(datasize[datasize['project']==projectname]['testsize'])
        
        print(len(train_idxs))
        print(trainsize)
        print(len(test_idxs))
        print(testsize)
        train_idxs = random.sample(train_idxs,trainsize)
        test_idxs = random.sample(test_idxs,testsize)

        with open("result", "a") as myfile:
            myfile.write('testing on project: ')
            myfile.write(projectname)
            myfile.write('\n')
            myfile.write('Actual train size: ')
            myfile.write(str(len(train_idxs)))
            myfile.write('\n')
            myfile.write('Actual test size: ')
            myfile.write(str(len(test_idxs)))
            myfile.write('\n')

    if args.equal2fullmodelsize:
        trainsize = 69706
        testsize = 29874
      
        train_idxs = random.sample(train_idxs,trainsize)
        test_idxs = random.sample(test_idxs,testsize)

        with open("result", "a") as myfile:
            myfile.write('testing on all projects ')
            myfile.write('\n') 
            myfile.write('Actual train size: ')
            myfile.write(str(len(train_idxs)))
            myfile.write('\n')
            myfile.write('Actual test size: ')
            myfile.write(str(len(test_idxs)))
            myfile.write('\n')

        

    X_train = [X[i] for i in train_idxs]
    y_train = [y[i] for i in train_idxs]
    X_test = [X[i] for i in test_idxs]
    y_test = [y[i] for i in test_idxs]

    comments_dict_flat_test = [comments_dict_flat[i] for i in test_idxs]

    do_model(args, X_train, y_train, X_test, y_test, args.epochs)

    


        

