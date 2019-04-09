import argparse, hashlib, csv, random, pandas, os
import numpy as np

from model_prediction_lib import create_committer_data_dict, create_comments_dict, \
                            train_test_split_idxs, get_index_tokenizer, build_data, \
                            trim_pad_sequences, flatten_comments_dict, mention_preprocess,\
                            remove_short_comments, Using_one_project, set_of_words_only,\
                            remove_set_of_words, randomizing_data, length_sequences,\
                            remove_non_committer
from model_prediction_lib import COMMIT_METRICS

from keras.models import Model, model_from_json, Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score

from util import load_cached_data, get_cache_path

MODEL_TYPES = ['classifier', 'commits']
DATA_TYPES = ['speaking', 'spoken_to']
VOCAB_SIZE = 15000 # This need to be consistant with the SDAE embedding
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ComputeAUC(Callback):

    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        print('\n\nComputing AUC for epoch %d' % epoch)
        y_score = self.model.predict(self.validation_data[0], batch_size=512)
        auc = float(roc_auc_score(self.validation_data[1], y_score)) * 100
        self.history.append(auc)
        print('\nAUC: %.5f%%' % auc)

class ComputeConfusionMatrix(Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('\nComputing confusion matrix for epoch %d' % epoch)
        y_preds_cls = self.model.predict_classes(self.validation_data[0], batch_size=512)
        conf_mat = confusion_matrix(self.validation_data[1], y_preds_cls)
        print('\nConfusion matrix:')
        conf_mat = confusion_matrix(self.validation_data[1], y_preds_cls)
        print(conf_mat)
        print('')

def do_classifier(args, X_train, y_train, X_test, y_test, epochs,tkr):
    embedding_length = 64

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)

    # Need to trim input data to some max length of tokens
    model = Sequential()
    if args.use_sdae_embedding:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        embedding_matrix = loaded_model.get_layer("embedding_1").get_weights()
        model.add(Embedding(input_dim=VOCAB_SIZE,
                            output_dim=embedding_length,
                            weights=embedding_matrix,
                            input_length=args.max_seq_len,
                            trainable=False))
    else:
        model.add(Embedding(input_dim=len(tkr.word_index) + 1,
                            output_dim=embedding_length,
                            input_length=args.max_seq_len,
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

    history = model.fit(X_train_np, y_train_np,
                        validation_data=(X_test_np, y_test_np),
                        epochs=epochs,
                        batch_size=512,
                        callbacks=[ComputeAUC(), ComputeConfusionMatrix()])

    y_score = model.predict(X_test_np, batch_size=512)
    auc = float(roc_auc_score(y_test_np, y_score)) * 100
    print('\nFinal AUC: %.5f%%' % auc)

    y_preds_cls = model.predict_classes(X_test_np, batch_size=512)
    conf_mat = confusion_matrix(y_test_np, y_preds_cls)
    print('\nFinal Confusion matrix:')
    conf_mat = confusion_matrix(y_test_np, y_preds_cls)
    print(conf_mat)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Formats data for LM input')
    argparser.add_argument('--base_comment_dir',
                           action='store',
                           help='Base directory of issue comments.',
                           default='../data/sample_46_projs_issues')
    argparser.add_argument('--base_commit_dir',
                           action='store',
                           help='Base directory of commits.',
                           default='../data/sample_46_projs_commits')
    argparser.add_argument('--comments_file',
                           action='store',
                           help='Comments file name.',
                           default='issue_comments_REPLACECODE.csv')
    argparser.add_argument('--model_type',
                           action='store',
                           help='Model type.',
                           choices=MODEL_TYPES,
                           default='classifier')
    argparser.add_argument('--data_type',
                           action='store',
                           help='Data type',
                           choices=DATA_TYPES,
                           default='speaking')
    argparser.add_argument('--top_perc',
                           action='store',
                           help='Top percent of the response metric to be the positive class',
                           type=int,
                           default=10)
    argparser.add_argument('--metric',
                           action='store',
                           help='Metric for the top percent.',
                           choices=COMMIT_METRICS,
                           type=str)
    argparser.add_argument('--max_seq_len',
                           action='store',
                           help='Maximum sequence (post) length to deal with. Larger will be trimmed, shorter will be padded.',
                           type=int,
                           default=32)
    argparser.add_argument('--epochs',
                           action='store',
                           help='Number of epochs.',
                           type=int,
                           default=2)
    argparser.add_argument('--min_sequence_len',
                           action='store',
                           help='Remove post that is shorter than the given length.',
                           type=int,
                           default=5)
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
    argparser.add_argument('--shuffle',
                           action='store_true',
                           help='Shuffle tokens of each post.',
                           default=False)
    argparser.add_argument('--keep_ws',
                           action='store_true',
                           help='Only keep a set of words.',
                           default=False)
    argparser.add_argument('--remove_ws',
                           action='store_true',
                           help='Remove a set of words.',
                           default=False)
    argparser.add_argument('--ws',
                           action='store',
                           help='Specify word set file path.'
                           default='../data/wordlist/closed_class_words.csv')
    argparser.add_argument('--use_sdae_embedding',
                           action='store_true',
                           help='Use embedding got from sdae-like model. Be careful with encoding dimention and verb size.',
                           default=False)
    argparser.add_argument('--only_committer',
                           action='store_true',
                           help='If only use data from committers.',
                           default=False)
    argparser.add_argument('--baseline',
                           action='store_true',
                           help='Randomize data')
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

    print('Model: ' + args.data_type)
    print('Min Sequence Length: %d' % args.min_sequence_len)
    print('Max Sequence Length: %d' % args.max_seq_len)

    # read in and cache committer and comments data
    committer_dict = create_committer_data_dict(args.base_commit_dir, args.top_perc, args.metric)
    h = hashlib.sha1()
    h_name = 'model_prediction_dl.py' + args.base_comment_dir + args.comments_file
    h.update(h_name.encode('utf-8'))
    comments_dict = create_comments_dict(args.base_comment_dir, committer_dict, args.comments_file, args.data_type, h)

    if args.one_project >= 0:
        print('Using one project')
        if args.test_project >= 0:
            raise Exception('Can\'t do cross project validation on 1 project')
        comments_dict,project_name = Using_one_project(args.one_project, comments_dict)

    print('Removing noise')
    comments_dict = remove_short_comments(comments_dict, args.min_sequence_len)

    if args.only_committer:
        print('Removing non-committers')
        comments_dict = remove_non_committer(comments_dict, committer_dict, args.data_type)

    if args.keep_ws:
        print('Keeping a set of words words')
        comments_dict = set_of_words_only(comments_dict, args.ws)
    elif args.remove_ccw:
        print('Removing a set of words')
        comments_dict = remove_set_of_words(comments_dict, args.ws)

    print('Splitting idxs')
    train_idxs, test_idxs, project_name = train_test_split_idxs(comments_dict, args.test_project)
    tkr = get_index_tokenizer(comments_dict, train_idxs)
    
    print('Building data')
    X, y, ids = build_data(comments_dict, committer_dict, tkr, args.model_type, args.data_type)

    if args.data_type == 'spoken_to':
        print('Trimming @mention')
        X = mention_preprocess(X, ids, args.max_seq_len)

    print('Converting texts to sequences')
    if args.use_sdae_embedding:
        tkrh = hashlib.sha1()
        tkrh.update(('model_sdae.py' + args.base_comment_dir + str(VOCAB_SIZE)  + 'tkr').encode('utf-8'))
        tkr = load_cached_data(tkrh)
        tkd_comments_indexed = []
        for ind, comment in enumerate(X):
            seq = []
            for word in comment.split(' '):
                add_index = tkr.word_index.get(word)
                if add_index:
                    add_index = tkr.word_index[word] if tkr.word_index[word] < VOCAB_SIZE - 1 else VOCAB_SIZE - 1
                    seq.append(add_index)
                else:
                    seq.append(VOCAB_SIZE-1)
            tkd_comments_indexed.append(seq)
        X = tkd_comments_indexed
    else:
        X = tkr.texts_to_sequences(X)

    if args.tokenlength_only:
        print('Converting all words into same token')
        X=length_sequences(X)
        print('Sequence length test')

    print('Trimming and padding to max_seq_len %d' % args.max_seq_len)
    X = trim_pad_sequences(X, args.max_seq_len, 0)

    print('Flattening comments dict')
    comments_dict_flat = flatten_comments_dict(comments_dict)

    print('Train size: %d' % len(train_idxs))
    print('Test size: %d' % len(test_idxs))

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
        print(project_name)
        trainsize = int(datasize[datasize['project']==project_name]['trainsize'])
        testsize = int(datasize[datasize['project']==project_name]['testsize'])
        
        print(len(train_idxs))
        print(trainsize)
        print(len(test_idxs))
        print(testsize)
        train_idxs = random.sample(train_idxs,trainsize)
        test_idxs = random.sample(test_idxs,testsize)

        print('testing on project: '+ project_name)
        print('Actual train size: %d' % len(train_idxs))
        print('Actual test size: %d' % len(test_idxs))

    if args.equal2fullmodelsize:
        trainsize = 69706
        testsize = 29874
      
        train_idxs = random.sample(train_idxs,trainsize)
        test_idxs = random.sample(test_idxs,testsize)

        with open("result", "a") as myfile:
            print('testing on all projects') 
            print('Actual train size: %d' % len(train_idxs))
            print('Actual test size: %d' % len(test_idxs))

    X_train = [X[i] for i in train_idxs]
    y_train = [y[i] for i in train_idxs]
    X_test = [X[i] for i in test_idxs]
    y_test = [y[i] for i in test_idxs]

    comments_dict_flat_test = [comments_dict_flat[i] for i in test_idxs]

    do_classifier(args, X_train, y_train, X_test, y_test, args.epochs, tkr)

    


        

