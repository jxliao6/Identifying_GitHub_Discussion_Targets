import argparse, csv, hashlib, itertools, multiprocessing, pickle, random, os,\
        sys
csv.field_size_limit(sys.maxsize)

import keras.backend as K
import numpy as np
import tensorflow as tf

from collections import OrderedDict, defaultdict
from joblib import Parallel, delayed
from keras.preprocessing.text import Tokenizer
from keras.models import Model, model_from_json
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, LSTM 
from keras.utils import plot_model
from keras.callbacks import Callback
from nltk.tokenize.casual import TweetTokenizer
from util import get_cache_path, load_cached_data 

START_TOKEN = '<<START>>'
END_TOKEN = '<<END>>'
AT_MENTION = '<<MENTION>>'
NUM_CORES = multiprocessing.cpu_count()

class post:
   # 'base class for a comment post'
   tk_body = ''
   login = ''
   mention_id = ''
   issue_num = 0
   datetime = ''
   project = ''
   predict = None

   def __init__(self, tk_body, login, mention_id, issue_num, datetime, project):
      self.tk_body = tk_body
      self.login = login
      self.mention_id = mention_id
      self.issue_num = issue_num
      self.datetime = datetime
      self.project = project

def tokenize_comments(base_dir, comments_file,hashh=None):
    tkd_data = None

    if hashh:
        tkd_data = load_cached_data(hashh)

    if tkd_data is None:
        hash_f = get_cache_path(hashh)
        with open(hash_f, 'wb') as pkl_f:
            tkd_data = defaultdict(dict)
            tk = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
            for i, (root, dirs, files) in enumerate(os.walk(base_dir)):
                if comments_file in files:
                    project = root.split('/')[-1]
                    print('Processing %s, number %d' % (project, i))
                    posts = []
                    with open(os.path.join(root, comments_file), 'r') as inf:
                        r = csv.DictReader(inf)
                        for row in r:
                            p = post(' '.join(list(tk.tokenize(row['body']))),
                                     row['login'],
                                     row['mention_login'],
                                     row['issue_num'],
                                     row['datetime'],
                                     project)
                            posts.append(p)

                    tkd_data[project] = posts
            pickle.dump(tkd_data, pkl_f)

    return tkd_data

def people_in_issue(base_dir, pred_file='issue_first_in.csv'):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for i, (root, dirs, files) in enumerate(os.walk(base_dir)):
        if pred_file in files:
            project = root.split('/')[-1]
            with open(os.path.join(root, pred_file), 'r') as inf:
                readCSV = csv.reader(inf, delimiter=',')
                for issue_num, login, datetime in readCSV:
                    data[project][issue_num][login] = datetime
    return data
                    
def pred_and_flat(tkd_data, pred_data):
    posts_flat = []
    projects = sorted(tkd_data.keys())
    for project in projects:
        posts = tkd_data[project]
        preds = pred_data[project]
        for post in posts:
            post.predict = [login for login in preds[post.issue_num].keys()
                            if preds[post.issue_num][login] > post.datetime]
            posts_flat.append(post)
    return posts_flat

def split_train_test(tkd_comments_flat, train_perc=0.7, seed=1337):
    idxs = list(range(len(tkd_comments_flat)))
    random.shuffle(idxs)
    train_idxs = idxs[ : int(train_perc * len(idxs))]
    test_idxs = idxs[int(train_perc * len(idxs)) : ]

    return train_idxs, test_idxs


def get_index_tokenizer(tkd_comments_flat):
    tkr = Tokenizer(num_words=None, filters='', lower=False, split=' ')
    # Need a start and end token, but don't want to construct target seqs
    # until AFTER we indexify. So, do it here, so the tokenizer has the mapping
    # for start and end tokens and we don't have to deal with it manually later.
    tkd_comments_wstartend = []
    for comment in tkd_comments_flat:
        tkd_comments_wstartend.append(START_TOKEN + ' ' + comment + ' ' + END_TOKEN)
    tkr.fit_on_texts(tkd_comments_wstartend)

    return tkr

def flatten_tk_comments(tkd_data):
    tk_bodies = []
    mention_ids = []

    # Need to guarantee an ordering
    projects = sorted(tkd_data.keys())
    for project in projects:
        comment_list, mentions = tkd_data[project]
        for ind, body in enumerate(comment_list):
            tk_bodies.append(body)
            mention_ids.append(mentions[ind])

    return tk_bodies, mention_ids

def remove_mentions(tkd_comments_flat, mention_ids_flat, predict_flat, max_seq_len):
    noise_comments = []
    mention_ids = []
    predicts = []
    for ind, post in enumerate(tkd_comments_flat):
        seq = post.split()
        name = '@' + mention_ids_flat[ind].split('-')[0]
        if name not in seq: # handle bad data -- need a better solution
            # print('-------' + name + ' not found---------')
            # print(seq)
            continue
        else:
            pos = seq.index(name)
            seq[pos] = AT_MENTION
            if pos > max_seq_len / 2:
                seq = seq[pos - int(max_seq_len / 2) :]
            noise_comments.append(' '.join(seq))
            mention_ids.append('@' + mention_ids_flat[ind])
            predicts.append(predict_flat[ind])

    return noise_comments, mention_ids, predicts


def trim_pad_sequences(sequences, max_len, pad_idx):
    newseqs = []
    for seq in sequences:
        if len(seq) > max_len:
            seq = seq[ : max_len]
        elif len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        newseqs.append(seq)

    return newseqs

def trim_pad_seq(seq, max_len, pad_idx=0):
    newseq = seq
    if len(newseq) > max_len:
        newseq = newseq[ : max_len]
    elif len(newseq) < max_len:
        newseq = newseq + [pad_idx]  * (max_len - len(newseq))

    return newseq

# For some reason, keras doesn't allow for replacing trimmed vocab (specified by Tokenizer nb_words)
# with 0 when converting text to sequences. So, we have to do it ourselves.
def convert_texts_to_sequences(tkd_comments_flat, mention_ids_flat, predict_flat, tkr, vocab_size, denoise):
    print('Converting texts to index sequences')

    mention_ids_indexed = []
    mention_dictionary = OrderedDict()
    for mention in mention_ids_flat:  
        if mention in mention_dictionary:  
            mention_dictionary[mention] = mention_dictionary[mention] + 1  
        else:  
            mention_dictionary[mention] = 1
    # print(list(mention_dictionary)[:5])
    mention_dictionary = OrderedDict(sorted(mention_dictionary.items(), key=lambda t: t[1], reverse=True))
    dict_keys = list(mention_dictionary.keys())
    # print(list(mention_dictionary)[:5])
    # print(dict_keys[0:5])
    for mention in mention_ids_flat:
        if (mention in dict_keys and mention_dictionary[mention] > 10) or not denoise:
            mention_ids_indexed.append(dict_keys.index(mention))
        else:
            mention_ids_indexed.append(-1)
            if mention in dict_keys:
                mention_dictionary.pop(mention)
                dict_keys.remove(mention)

    if not vocab_size:
        return tkr.texts_to_sequences(tkd_comments_flat), mention_ids_indexed

    tkd_comments_indexed = []
    tkd_predict_flat = []

    for ind, comment in enumerate(tkd_comments_flat):
        if mention_ids_indexed[ind] < 0:
            continue
        seq = []
        for word in comment.split(' '):
            add_index = tkr.word_index.get(word)
            if add_index:
                add_index = tkr.word_index[word] if tkr.word_index[word] < vocab_size-1 else vocab_size-1
                seq.append(add_index)
        tkd_comments_indexed.append(seq)
        tkd_predict_flat.append(predict_flat[ind])
    mention_ids_indexed = [mention_id for mention_id in mention_ids_indexed if mention_id >= 0]

    return tkd_comments_indexed, mention_ids_indexed, tkd_predict_flat, len(mention_dictionary)+1, mention_dictionary


def fill_decoder_target_data(i, t, tok_idx, decoder_target_data):
    if t > 0:
        decoder_target_data[i, t - 1, tok_idx] = 1


# For this to work, decoder/encoder input data should be standard for embedding layers,
# but decoder target data needs to be one-hot encoded
def construct_input_target_data(tkd_comments_indexed, tkr, max_seq_len, sample_size, vocab_size):
    if not sample_size:
        sample_size = len(tkd_comments_indexed)

    input_seqs = [trim_pad_seq(tkd_comments_indexed[i], max_seq_len) for i in range(sample_size)]
    target_seqs = [[tkr.word_index[START_TOKEN]] + trim_pad_seq(tkd_comments_indexed[i], max_seq_len) + [tkr.word_index[END_TOKEN]] for i in range(sample_size)]

    num_encoder_tokens = vocab_size if vocab_size else len(tkr.word_index)
    num_decoder_tokens = vocab_size if vocab_size else len(tkr.word_index)
    max_encoder_seq_len = max([len(seq) for seq in input_seqs])
    max_decoder_seq_len = max([len(seq) for seq in target_seqs])

    print('Decoder target shape: %s' % ((len(input_seqs), max_decoder_seq_len, num_decoder_tokens), ))
    print('Allocating decoder_target_data')
    decoder_target_data = np.zeros(
        (len(input_seqs), max_decoder_seq_len, num_decoder_tokens),
        dtype='float32')

    print('Filling decoder_target_data')
    print('Maximum iteration count: %d' % (len(input_seqs) * max_decoder_seq_len * num_decoder_tokens, ))

    Parallel(n_jobs=NUM_CORES, backend='threading')(delayed(fill_decoder_target_data)(i, t, tok_idx, decoder_target_data) for i, target_seq in enumerate(target_seqs) for t, tok_idx in enumerate(target_seq))
    # for i, target_seq in enumerate(target_seqs):
    #     for t, tok_idx in enumerate(target_seq):
    #         if t > 0:
    #             decoder_target_data[i, t - 1, tok_idx] = 1

    return num_encoder_tokens, num_decoder_tokens, np.array(input_seqs), np.array(target_seqs), decoder_target_data

def construct_sdae(num_encoder_tokens, num_decoder_tokens, encoding_dim):
    print('Constructing model')
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, encoding_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(encoding_dim, return_state=True)(x)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    x = Embedding(num_decoder_tokens, encoding_dim)(decoder_inputs)
    x = LSTM(encoding_dim, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_sample_partitions(tkd_comments_indexed, train_idxs, test_idxs, sample_size):
    max_iter = len(train_idxs)
    partition = {'train': [], 'test': []}
    for idx in range(sample_size, max_iter, sample_size):
        partition['train'].append((idx - sample_size, idx))
    if max_iter % sample_size != 0:
        partition['train'].append((max_iter - max_iter % sample_size, max_iter - 1))
    partition['test'].append((0, len(test_idxs)))

    return partition


def fetch_batch(tkd_comments_indexed, mention_ids_indexed, tkr, idxs, max_seq_len, vocab_size):
    input_seqs = [trim_pad_seq(tkd_comments_indexed[i], max_seq_len) for i in idxs]
    # target_seqs = [[tkr.word_index[START_TOKEN]] + trim_pad_seq(tkd_comments_indexed[i], max_seq_len) + [tkr.word_index[END_TOKEN]] for i in idxs]
    target_seqs = [[vocab_size] + [mention_ids_indexed[i]] for i in idxs]
    tmp = [[tkr.word_index[START_TOKEN]] for i in idxs]
    # mentions = [mention_ids_indexed[i] for i in idxs]

    num_decoder_tokens = vocab_size
    max_encoder_seq_len = max([len(seq) for seq in input_seqs])
    max_decoder_seq_len = max([len(seq) for seq in target_seqs])

    decoder_target_data = np.zeros(
        (len(input_seqs), 1, num_decoder_tokens),
        dtype='float32')

    Parallel(n_jobs=NUM_CORES, backend='threading')(delayed(fill_decoder_target_data)(i, t, tok_idx, decoder_target_data) for i, target_seq in enumerate(target_seqs) for t, tok_idx in enumerate(target_seq))
    # Parallel(n_jobs=NUM_CORES, backend='threading')(delayed(fill_decoder_target_data)(i, 1, mentions[i], decoder_target_data) for i, target_seq in enumerate(target_seqs))

    return np.array(input_seqs), np.array(tmp), decoder_target_data


def data_generator(tkd_comments_indexed, mention_ids_indexed, tkr, idxs, max_seq_len, vocab_size, batch_size):
    while 1:
        max_iter = int(len(idxs) / batch_size)
        for i in range(max_iter):
            batch_idxs = idxs[i * batch_size : (i + 1) * batch_size]
            encoder_input_data, decoder_input_data, decoder_target_data = \
                fetch_batch(tkd_comments_indexed, mention_ids_indexed, tkr, batch_idxs, max_seq_len, vocab_size)

            yield [encoder_input_data, decoder_input_data], decoder_target_data

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Recommandation system using autoencoder')
    argparser.add_argument('--base_comment_dir',
                           action='store',
                           help='Base directory of issue comments',
                           default='../data/sample_48_projs_issues')
    argparser.add_argument('--comments_file',
                           action='store',
                           help='name of the file containing posts',
                           default='issue_comments_mentioned_committer_NOCODE_FULLBODY.csv')
    argparser.add_argument('--max_seq_len',
                           action='store',
                           help='Maximum sequence (post) length to deal with. \
                                Larger will be trimmed, shorter will be padded.',
                           type=int,
                           default=32)
    argparser.add_argument('--epochs',
                           action='store',
                           help='Number of epochs for first training',
                           type=int,
                           default=20)
    argparser.add_argument('--epochs_2',
                           action='store',
                           help='Number of epochs for retraining',
                           type=int,
                           default=10)
    argparser.add_argument('--batch_size',
                           action='store',
                           help='Batch size',
                           type=int,
                           default=1024)
    argparser.add_argument('--encoding_dim',
                           action='store',
                           help='Dimension of encoding (latent dimension)',
                           type=int,
                           default=200)
    argparser.add_argument('--vocab_size',
                           action='store',
                           help='Max vocab size to consider. Default is 15,000',
                           type=int,
                           default=15000)
    argparser.add_argument('--denoise',
                           help='Remove posts with few tokens',
                           action='store_true',
                           default=False)
    args = argparser.parse_args()

    h = hashlib.sha1()
    h.update(('sdae_responsive_taggee.py' + args.base_comment_dir + args.comments_file)
             .encode('utf-8'))

    tkd_data = tokenize_comments(args.base_comment_dir, hashh=h,
                                 comments_file=args.comments_file)
    pred_data = people_in_issue(args.base_comment_dir)
    tkd_posts_flat = pred_and_flat(tkd_data, pred_data)

    random.seed(1024)
    random.shuffle(tkd_posts_flat)

    tkd_comments_flat = []
    mention_ids_flat = []
    predict_flat = []
    for post in tkd_posts_flat:
        tkd_comments_flat.append(post.tk_body)
        mention_ids_flat.append(post.mention_id)
        predict_flat.append(post.predict)

    tkd_comments_flat, mention_ids_flat, predict_flat = \
        remove_mentions(tkd_comments_flat, mention_ids_flat, predict_flat, args.max_seq_len)

    tkd_comments_flat_train = []
    tkd_comments_flat_train2 = []
    mention_ids_flat_train = []
    mention_ids_flat_train2 = []
    predict_flat_train = []
    predict_flat_train2 = []
    for ind, mention_id in enumerate(mention_ids_flat):
        if mention_id[1:] in predict_flat[ind]:       
        	tkd_comments_flat_train2.append(tkd_comments_flat[ind])
        	mention_ids_flat_train2.append(mention_id)
        	predict_flat_train2.append(predict_flat[ind])
        else:
        	tkd_comments_flat_train.append(tkd_comments_flat[ind])
        	mention_ids_flat_train.append(mention_id)
        	predict_flat_train.append(predict_flat[ind])
    tkd_comments_flat_test = tkd_comments_flat_train2[ : int(0.2 * len(tkd_comments_flat_train2))]
    tkd_comments_flat_train2 = tkd_comments_flat_train2[int(0.2 * len(tkd_comments_flat_train2)) : ]
    mention_ids_flat_test = mention_ids_flat_train2[ : int(0.2 * len(mention_ids_flat_train2))]
    mention_ids_flat_train2 = mention_ids_flat_train2[int(0.2 * len(mention_ids_flat_train2)) : ]
    predict_flat_test = predict_flat_train2[ : int(0.2 * len(predict_flat_train2))]
    predict_flat_train2 = predict_flat_train2[int(0.2 * len(predict_flat_train2)) : ]

    tkd_comments_flat = tkd_comments_flat_train + tkd_comments_flat_train2 + tkd_comments_flat_test
    mention_ids_flat = mention_ids_flat_train + mention_ids_flat_train2 + mention_ids_flat_test
    predict_flat = predict_flat_train + predict_flat_train2 + predict_flat_test
    len1 = len(tkd_comments_flat_train)
    len2 = len(tkd_comments_flat_train) + len(tkd_comments_flat_train2)
    len3 = len(tkd_comments_flat_train) + len(tkd_comments_flat_train2) + len(tkd_comments_flat_test)
    train_idxs1 = list(range(len1))
    train_idxs2 = list(range(len1, len2))
    test_idxs = list(range(len2, len3))

    # test
    print((len1,len2,len3))
    
    tkr = get_index_tokenizer(tkd_comments_flat)

    tkd_comments_indexed, mention_ids_indexed, predict_flat, vocab_size, mention_dictionary = \
        convert_texts_to_sequences(tkd_comments_flat, mention_ids_flat, predict_flat, tkr, 
        	                       args.vocab_size, args.denoise)

    mention_dictionary = list(mention_dictionary)

    model = construct_sdae(args.vocab_size, vocab_size, args.encoding_dim)
    print('Fitting model')


    encoder_input_data, decoder_input_data, decoder_target_data = \
        fetch_batch(tkd_comments_indexed, mention_ids_indexed, tkr, train_idxs1, args.max_seq_len, vocab_size)
    encoder_input_data2, decoder_input_data2, decoder_target_data2 = \
        fetch_batch(tkd_comments_indexed, mention_ids_indexed, tkr, train_idxs2, args.max_seq_len, vocab_size)
    encoder_input_data_val, decoder_input_data_val, decoder_target_data_val = \
        fetch_batch(tkd_comments_indexed, mention_ids_indexed, tkr, test_idxs, args.max_seq_len, vocab_size)
    predict_flat_test = [predict_flat[i] for i in test_idxs]
    tkd_comments_flat_test = [tkd_comments_flat[i] for i in test_idxs]
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, 
        validation_data=([encoder_input_data_val, decoder_input_data_val], decoder_target_data_val),
        batch_size=512, epochs=args.epochs)

    with open('responsive_taggee_predict1', 'w') as file:
        reverse_word_map = dict(map(reversed, tkr.word_index.items()))
        pred = model.predict([encoder_input_data_val, decoder_input_data_val])
        mention_dictionary = list(mention_dictionary)
  
        for i in range(len(encoder_input_data_val)):
            for token in encoder_input_data_val[i]:
                if token == 0:
                    file.write('<padding> ')
                elif token == args.vocab_size-1:
                    file.write('<infrequent> ')
                else:
                    file.write(reverse_word_map[token])
                    file.write(' ')
            file.write('\n')
            token = np.argmax(decoder_target_data_val[i][0])
            file.write('Actual: ')
            file.write(mention_dictionary[token])
            file.write(' ')
            token2 = np.argmax(pred[i][0])
            file.write('Predict: ')
            file.write(mention_dictionary[token2])
            file.write(' ')
            sort = np.argsort(pred[i][0])[::-1]
            rank = np.nonzero(sort == token)[0] + 1
            file.write('Rank: ')
            file.write(str(rank[0]))
            file.write(' ')
            file.write('\n')
            file.write('Later in: ')
            for p in predict_flat_test[i]:
                file.write(p)
                file.write(' ')
            file.write('\n')

    model.fit([encoder_input_data2, decoder_input_data2], decoder_target_data2, 
        validation_data=([encoder_input_data_val, decoder_input_data_val], decoder_target_data_val),
        batch_size=512, epochs=args.epochs_2)

    with open('responsive_taggee_predict2', 'w') as file:
        pred = model.predict([encoder_input_data_val, decoder_input_data_val])
  
        for i in range(len(encoder_input_data_val)):
            for token in encoder_input_data_val[i]:
                if token == 0:
                    file.write('<padding> ')
                elif token == args.vocab_size-1:
                    file.write('<infrequent> ')
                else:
                    file.write(reverse_word_map[token])
                    file.write(' ')
            file.write('\n')
            token = np.argmax(decoder_target_data_val[i][0])
            file.write('Actual: ')
            file.write(mention_dictionary[token])
            file.write(' ')
            token2 = np.argmax(pred[i][0])
            file.write('Predict: ')
            file.write(mention_dictionary[token2])
            file.write(' ')
            sort = np.argsort(pred[i][0])[::-1]
            rank = np.nonzero(sort == token)[0] + 1
            file.write('Rank: ')
            file.write(str(rank[0]))
            file.write(' ')
            file.write('\n')
            file.write('Later in: ')
            for p in predict_flat_test[i]:
                file.write(p)
                file.write(' ')
            file.write('\n')
