import argparse, hashlib, csv, random
import numpy as np

from format_lm_input_write import create_committer_data_dict, create_comments_dict, \
                            train_test_split_idxs, get_index_tokenizer, build_data, \
                            trim_pad_sequences, flatten_comments_dict, mention_preprocess,\
                            remove_short_comments, Using_one_project, closed_class_only,\
                            remove_closed_class

from format_lm_input_write import COMMIT_METRICS

from keras.models import Model, model_from_json, Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score

from util import load_cached_data, get_cache_path

DATA_TYPES = ['speaking', 'spoken_to']
VOCAB_SIZE = 15000
RESULT_FILE = 'lm_result'
MODEL_TYPES = ['classifier', 'commits']

class ComputeAUC(Callback):

    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        print('\n\nComputing AUC for epoch %d' % epoch)
        y_score = self.model.predict(self.validation_data[0], batch_size=512)
        auc = float(roc_auc_score(self.validation_data[1], y_score)) * 100
        self.history.append(auc)
        print('\nAUC: %.5f%%' % auc)
        with open(RESULT_FILE,'a') as file:
            file.write(' ' + str(auc));

class ComputeConfusionMatrix(Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('\nComputing confusion matrix for epoch %d' % epoch)
        y_preds_cls = self.model.predict_classes(self.validation_data[0], batch_size=512)
        conf_mat = confusion_matrix(self.validation_data[1], y_preds_cls)
        print('\nConfusion matrix:')
        conf_mat = confusion_matrix(self.validation_data[1], y_preds_cls)
        print(conf_mat)
        print('')

def do_classifier(args, X_train, y_train, X_test, y_test, epochs, tkr):
    embedding_length = 64

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)

    if args.use_sdae_embedding:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")

    # Need to trim input data to some max length of tokens
    model = Sequential()
    if args.use_sdae_embedding:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        embedding_matrix = loaded_model.get_layer("embedding_1").get_weights();
        print((np.array(embedding_matrix)).shape)
        model.add(Embedding(input_dim=VOCAB_SIZE,
                            output_dim=embedding_length,
                            weights=embedding_matrix,
                            input_length=args.max_seq_len,
                            trainable=False))
    else:
        model.add(Embedding(input_dim=15000,
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
                        callbacks=[ComputeAUC(), ComputeConfusionMatrix()]
                        )

    y_score = model.predict(X_test_np, batch_size=512)
    auc = float(roc_auc_score(y_test_np, y_score)) * 100
    print('\nFinal AUC: %.5f%%' % auc)

    with open(RESULT_FILE,'a') as file:
        file.write('\n\n')

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
                           default='../data/sample_48_projs_issues')
    argparser.add_argument('--base_commit_dir',
                           action='store',
                           help='Base directory of commits.',
                           default='../data/sample_48_projs_commits')
    argparser.add_argument('comments_file',
                           action='store',
                           help='Comments file name.',
                           default='issue_comments_REPLACECODE.csv')
    argparser.add_argument('--model_type',
                           action='store',
                           help='Model type.',
                           choices=MODEL_TYPES,
                           default='classifier')
    argparser.add_argument('--top_perc',
                           action='store',
                           help='Top percent of the response metric to be the \
                           positive class.',
                           type=int,
                           default=10)
    argparser.add_argument('--data_type',
                           action='store',
                           help='Data type',
                           choices=DATA_TYPES,
                           default='spoken_to')
    argparser.add_argument('--metric',
                           action='store',
                           help='Metric for the top percent.',
                           choices=COMMIT_METRICS,
                           type=str,
                           default='commits')
    argparser.add_argument('--max_seq_len',
                           action='store',
                           help='Maximum sequence(post) length to deal with. \
                           Larger will be trimmed, shorter will be padded.',
                           type=int,
                           default=32)
    argparser.add_argument('--epochs',
                           action='store',
                           help='Number of epochs.',
                           type=int,
                           default=10)
    argparser.add_argument('--noise_len',
                           action='store',
                           help='Post has less than this number of token is \
                           considered noise and will be removed.',
                           type=int,
                           default=0)
    argparser.add_argument('--test_project',
                           action='store',
                           help='Test on this project and train on all others',
                           type=int,
                           default=-1)
    argparser.add_argument('--one_project',
                           action='store',
                           help='Only use data in this project.',
                           type=int,
                           default=-1)
    argparser.add_argument('--shuffle',
                           action='store_true',
                           help='Shuffle tokens of each post.')
    argparser.add_argument('--keep_ccw',
                           action='store_true',
                           help='Only keep closed class words')
    argparser.add_argument('--remove_ccw',
                           action='store_true',
                           help='Remove closed class words')
    argparser.add_argument('--use_sdae_embedding',
                           action='store_true',
                           help='Use embedding got from sdae-like model. Be careful with encoding dimention and verb size.')

    args = argparser.parse_args()

    with open(RESULT_FILE,'a') as file:
        file.write('model type: ' + args.model_type + '\n')
        file.write('test type: ')
        if args.use_sdae_embedding:
            file.write('Using sdae embedding\n')
        if args.test_project >= 0:
            file.write('cross project\n')
        elif args.one_project >= 0:
            file.write('within one project\n')

    # read in and cache committer and comments data
    committer_dict = create_committer_data_dict(args.base_commit_dir, 
    	args.top_perc, args.metric)
    h = hashlib.sha1()
    h_name = 'language_status.py' + args.base_comment_dir + args.comments_file
    h.update(h_name.encode('utf-8'))
    comments_dict = create_comments_dict(args.base_comment_dir, committer_dict,
    	args.comments_file, args.data_type, h)

    if args.one_project >= 0:
        print('Using one project')
        if args.test_project >= 0:
            raise Exception('Can\'t do cross project validation on 1 project')
        comments_dict = Using_one_project(args.one_project, comments_dict)

    print('Removing noise')
    comments_dict = remove_short_comments(comments_dict, args.noise_len)

    if args.keep_ccw:
      print('Keeping only closed class words')
      comments_dict = closed_class_only(comments_dict)
    elif args.remove_ccw:
      print('Removing closed class words')
      comments_dict = remove_closed_class(comments_dict)

    print('Splitting idxs')
    if args.test_project == -1:
        train_idxs, test_idxs = train_test_split_idxs(comments_dict, args.test_project)
    else:
        train_idxs, test_idxs,projectname = train_test_split_idxs(comments_dict, args.test_project)
    tkr = get_index_tokenizer(comments_dict, train_idxs)

    print('Building data')
    X, y, ids = build_data(comments_dict, committer_dict, tkr, args.model_type,args.data_type)

    if args.data_type == 'spoken_to':
        print('Trimming @mention')
        X = mention_preprocess(X, ids, args.max_seq_len)

    print('Converting texts to sequences')
    if args.use_sdae_embedding:
        tkrh = hashlib.sha1()
        tkrh.update(('sdae.py' + args.base_comment_dir + str(VOCAB_SIZE)  + 'tkr').encode('utf-8'))
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

    # pad idx is 0
    print('Trimming and padding to max_seq_len %d' % args.max_seq_len)
    X = trim_pad_sequences(X, args.max_seq_len, 0)

    print('Flattening comments dict')
    comments_dict_flat = flatten_comments_dict(comments_dict)

    if args.shuffle:
        print("shuffling data")
        for ind, _ in enumerate(X):
            random.shuffle(X[ind])

    X_train = [X[i] for i in train_idxs]
    y_train = [y[i] for i in train_idxs]
    X_test = [X[i] for i in test_idxs]
    y_test = [y[i] for i in test_idxs]

    comments_dict_flat_test = [comments_dict_flat[i] for i in test_idxs]

    do_classifier(args, X_train, y_train, X_test, y_test, args.epochs, tkr)

    


        

