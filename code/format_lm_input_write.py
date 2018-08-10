import argparse, os, csv, sys, hashlib, pickle, random, string
csv.field_size_limit(sys.maxsize)

from nltk.tokenize.casual import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from util import load_cached_data, get_cache_path
import pickle

COMMIT_METRICS = ['additions', 'deletions', 'total', 'commits']

def tokenize_comments(f):
    rows = []
    tk = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
    project = f.split('/')[-2]
    with open(f, 'r') as inf:
        r = csv.DictReader(inf)
        for row in r:
            tk_body = ' '.join(list(tk.tokenize(row['body'])))
            rows.append({'tokenized_body': tk_body,
                         'issue_num': row['issue_num'],
                         'datetime': row['datetime'],
                         'login': row['login'],
                         'close_date': row['close_date'],
                         'project': project})
    return rows

# Form of committer data dict:
# d[project][author (login)] = dict of key value pairs for relevant columns e.g.
#                    total number of commits, total number of additions, etc
def create_committer_data_dict(base_dir, top_perc, metric):
    data_dict = {}
    for root, dirs, files in os.walk(base_dir):
        if 'commits.csv' in files:
            author_data_dict = defaultdict(lambda: defaultdict(int))
            project = root.split('/')[-1]

            with open(os.path.join(root, 'commits.csv'), 'r') as inf:
                r = csv.DictReader(inf)
                for row in r:
                    author = row['author']
                    author_data_dict[author]['additions'] += int(row['additions'])
                    author_data_dict[author]['deletions'] += int(row['deletions'])
                    author_data_dict[author]['total'] += int(row['total'])
                    author_data_dict[author]['commits'] += 1

            # Calculate which users are in the top_perc for metric
            metrics = []
            authors = []
            for author, author_data in author_data_dict.items():
                authors.append(author)
                metrics.append(author_data[metric])
            # Happens if data gathering had an error and the csv is empty
            if len(metrics) == 0:
                continue
            # Sort by metrics
            metrics, authors = zip(*sorted(zip(metrics, authors)))
            # Figure out which authors are in top_perc
            for author, author_data in author_data_dict.items():
                author_data['top_n_class'] = authors.index(author) >= len(metrics) - top_perc / 100 * len(metrics)
                author_data['top_n_rank'] = authors.index(author)
                author_data['top_n_rank_cutoff'] = len(metrics) - top_perc / 100 * len(metrics)
                author_data['top_n_metric_value'] = metrics[authors.index(author)]

            data_dict[project] = author_data_dict

    return data_dict


def random_word(length):
    word = []
    for _ in range(length):
        word.append(random.choice(string.ascii_letters))

    return ''.join(word)

# Form of comments_dict
# d[project] = list of dict of key value pairs for each comment
def create_comments_dict(base_dir, committer_dict, comments_file, data_type, hashh=None):
    comments_dict = None
    if data_type == 'speaking':
        login = 'login'
    else:
        login = 'mention_login'

    if hashh:
        comments_dict = load_cached_data(hashh)
    if comments_dict is None:
        hash_f = get_cache_path(hashh)
        with open(hash_f, 'wb') as pkl_f:
            comments_dict = defaultdict(dict)
            tk = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
            project_num = 0
            for root, dirs, files in os.walk(base_dir):
                # project_num += 1
                # if project_num == 3:
                #     break
                if comments_file in files:
                    project = root.split('/')[-1]
                    # Can happen if we're missing a commits file
                    if project not in committer_dict:
                        continue
                    print('Processing %s' % project)
                    developerlist = list(committer_dict[project].keys())
                    rows = []
                    with open(os.path.join(root, comments_file), 'r') as inf:
                        r = csv.DictReader(inf)
                        for row in r:
                            seq = list(tk.tokenize(row['body']))
                            if row[login] not in developerlist:
                                    continue
                            if data_type == 'spoken_to':
                                name = '@' + row[login].split('-')[0] # some data are not formatted correctly
                                if name not in seq: # handle bad data -- need a better solution
                                    continue
                                else:
                                    pos = seq.index(name)
                                    seq[pos] = '@'
                            if row[login] not in developerlist:
                                continue
                            tk_body = ' '.join(seq)
                            row['tk_body'] = tk_body
                            row['top_n_class'] = committer_dict[project][row[login]]['top_n_class']
                            row['top_n_rank'] = committer_dict[project][row[login]]['top_n_rank']
                            row['top_n_rank_cutoff'] = committer_dict[project][row[login]]['top_n_rank_cutoff']
                            row['top_n_metric_value'] = committer_dict[project][row[login]]['top_n_metric_value']
                            row['project'] = project
                            row.pop('body')
                            rows.append(row)

                    comments_dict[project] = rows

            pickle.dump(comments_dict, pkl_f)

    return comments_dict

def Using_one_project(using_project, comments_dict):
    new_dict = defaultdict(dict)
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    chosen_projectname = projects[using_project]

    for ind,one_projectname in enumerate(projects):
        if ind == using_project:
            comment_list = comments_dict[one_projectname]
            new_dict[one_projectname] = comment_list
            with open("result", "a") as myfile:
                myfile.write('using project: ')
                myfile.write(one_projectname)
                myfile.write('\n')

    return new_dict,chosen_projectname



def randomizing_data(comments_dict):
    new_dict = defaultdict(dict)
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())

    for project in projects:
        comment_list = comments_dict[project]
        random_words = []
        for ind, d in enumerate(comment_list):
            for token in d['tk_body'].split():
                random_words.append(random_word(len(token)))
            comment_list[ind]['tk_body'] = ' '.join(random_words)
        new_dict[project] = comment_list
        # for d in new_dict[project]:
        #     print(d['tk_body'])
        # print('--------------------')

    return new_dict

def remove_short_comments(comments_dict, valid_len):
    clean_dict = defaultdict(dict)

    for project in comments_dict:
        comment_list = comments_dict[project]
        clean_comments = []
        for d in comment_list:
            if len(d['tk_body'].split()) > valid_len:
                clean_comments.append(d)
        clean_dict[project] = clean_comments

    return clean_dict

def get_index_tokenizer(comments_dict, train_idxs):
    tk_bodies = []
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    for project in projects:
        comment_list = comments_dict[project]
        for d in comment_list:
            tk_bodies.append(d['tk_body'])

    tkr = Tokenizer(num_words=None, filters='', lower=False, split=' ')
    tkr.fit_on_texts([tk_bodies[i] for i in train_idxs])

    return tkr

# Returns train and test split idxs
def train_test_split_idxs(comments_dict, test_project, train_perc=0.7, seed=1337):
    tk_bodies = []
    random.seed(seed)
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    testprojectname = projects[test_project]
    
    if test_project == -1: # no cross project validation
        for project in projects:
            comment_list = comments_dict[project]
            for d in comment_list:
                tk_bodies.append(d['tk_body'])
        idxs = list(range(len(tk_bodies)))
        random.shuffle(idxs)
        train_idxs = idxs[ : int(train_perc * len(idxs))]
        test_idxs = idxs[int(train_perc * len(idxs)) : ]

        return train_idxs, test_idxs
    else: # cross project validation: test project #test_project
        with open("result", "a") as myfile:
             myfile.write('testing on project: ')
             myfile.write(projects[test_project])
             myfile.write('\n')
        train_idxs = []
        test_idxs = []
        index = 0
        for ind, project in enumerate(projects):
            comment_list = comments_dict[project]
            for d in comment_list:
                if ind == test_project:
                    test_idxs.append(index)
                else:
                    train_idxs.append(index)
                index += 1
    return train_idxs, test_idxs,testprojectname

def mention_preprocess(sequences, ids, max_len):
    newseqs = []

    for seq in sequences:
        seq = seq.split(' ');
        try:
            pos_at = seq.index('@')
            if pos_at > max_len / 2: 
                seq = seq[pos_at - int(max_len / 2) :]
        except Exception as e:
            print('-----@_not_found----')
        seq = ' '.join(seq)
        newseqs.append(seq)

    return newseqs

def build_data(comments_dict, committer_dict, tokenizer, model_type, data_type):
    X, y, ids = [], [], []
    if data_type == 'speaking':
        login = 'login'
    else:
        login = 'mention_login'

    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    for project in projects:
        comment_list = comments_dict[project]
        for d in comment_list:
            X.append(d['tk_body'])
            ids.append(d[login])
            if model_type == 'classifier':
                yval = committer_dict[project][d[login]]['top_n_class']
            else:
                yval = committer_dict[project][d[login]][model_type]
            if yval == True:
                yval = 1
            elif yval == False:
                yval = 0
            y.append(yval)

    return X, y, ids



def trim_pad_sequences(sequences, max_len, pad_idx):
    newseqs = []
    for seq in sequences:
        if len(seq) > max_len:
            seq = seq[ : max_len]
        elif len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        newseqs.append(seq)

    return newseqs

def length_sequences(sequences):
    newseqs = []
    for seq in sequences:
        length = len(seq)
        newseqs.append([0]*length)

    return newseqs

def get_ccw_set(wordfile):
    s = set()

    with open(wordfile, "r") as myfile:
        for line in myfile.readlines():
            s.add(line.split()[0])  

    return s

def closed_class_only(comments_dict,wordfile):
    new_dict = defaultdict(dict)
    ccw = get_ccw_set(wordfile)

    for project in comments_dict.keys():
        comment_list = comments_dict[project]
        for ind,d in enumerate(comment_list):
            seq = []
            for token in d['tk_body'].split(' '):
                if token in ccw:
                    seq.append(token)
                else: # use <ocw> to replace open class words
                    seq.append('<ocw>')
            comment_list[ind]['tk_body'] = ' '.join(seq)
        new_dict[project] = comment_list

    return new_dict

def remove_closed_class(comments_dict,wordfile):
    new_dict = defaultdict(dict)
    ccw = get_ccw_set(wordfile)

    for project in comments_dict.keys():
        comment_list = comments_dict[project]
        for ind,d in enumerate(comment_list):
            seq = []
            for token in d['tk_body'].split(' '):
                if token in ccw:
                    seq.append('<ccw>')
                else:
                    seq.append(token)
            comment_list[ind]['tk_body'] = ' '.join(seq)
        new_dict[project] = comment_list

    return new_dict

def flatten_comments_dict(comments_dict):
    flattened = []
    projects = sorted(comments_dict.keys())
    for project in projects:
        comment_list = comments_dict[project]
        for d in comment_list:
            flattened.append(d)

    return flattened

def flatten_committer_dict(committer_dict):
    flattened = []
    projects = sorted(committer_dict.keys())
    for project in projects:
        c_dict = committer_dict[project]
        for user, data in c_dict.items():
            wrow = {'login': user, 'project': project}
            wrow.update(data)
            flattened.append(wrow)

    return flattened

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
    argparser.add_argument('top_perc',
                           action='store',
                           help='Top percent of the response metric to be the positive class',
                           type=int)
    argparser.add_argument('metric',
                           action='store',
                           help='Which metric do we want for the top percent?',
                           choices=COMMIT_METRICS,
                           type=str)

    args = argparser.parse_args()

    committer_dict = create_committer_data_dict(args.base_commit_dir, args.top_perc, args.metric)
    # h = hashlib.sha1()
    # h.update(('format_lm_input.py' + args.base_comment_dir + args.base_commit_dir + args.comments_file).encode('utf-8'))
    # comments_dict = create_comments_dict(args.base_comment_dir, committer_dict, args.comments_file, h)

    # train_idxs, test_idxs = train_test_split_idxs(comments_dict)
    # tkr = get_index_tokenizer(comments_dict, train_idxs)
    
    # X, y = build_data(comments_dict, committer_dict, tkr)

    committer_dict_flat = flatten_committer_dict(committer_dict)
    with open('./data/committer_dict_data.csv', 'w') as outf:
        w = csv.DictWriter(outf, fieldnames=committer_dict_flat[0].keys())
        w.writeheader()
        for d in committer_dict_flat:
            w.writerow(d)
