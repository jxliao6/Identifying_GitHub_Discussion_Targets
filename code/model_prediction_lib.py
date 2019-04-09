import argparse, os, csv, sys, hashlib, pickle, random, string
csv.field_size_limit(sys.maxsize)

from nltk.tokenize.casual import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from util import load_cached_data, get_cache_path
import pickle

COMMIT_METRICS = ['additions', 'deletions', 'total', 'commits']

# Form of committer data dict:
# d[project][author (login)] = dict of key value pairs for relevant columns e.g.
# total number of commits, total number of additions, etc
def create_committer_data_dict(base_dir, top_perc, metric):
    data_dict = {}
    for root, dirs, files in os.walk(base_dir):
        if 'commits.csv' not in files:
            continue
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

# Form of comments_dict
# d[project] = list of dict of key value pairs for each comment
def create_comments_dict(base_dir, committer_dict, comments_file, data_type, hashh=None):
    comments_dict = None
    # Specify user id key based on data type(speaking vs. spoken_to)
    login = 'login' if data_type == 'speaking' else 'mention_login'

    if hashh:
        comments_dict = load_cached_data(hashh)
    if comments_dict is None:
        hash_f = get_cache_path(hashh)
        with open(hash_f, 'wb') as pkl_f:
            comments_dict = defaultdict(dict)
            tk = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
            project_num = 0

            for root, dirs, files in os.walk(base_dir):
                project = root.split('/')[-1]

                if comments_file not in files:
                    continue
                # Can happen if we're missing a commits file
                if project not in committer_dict:
                    print('commits file missing in' + project)
                    continue

                print('Processing %s' % project)
                # User who have commits
                developerlist = list(committer_dict[project].keys())
                rows = []
                # Build metrics dictionary
                with open(os.path.join(root, comments_file), 'r') as inf:
                    r = csv.DictReader(inf)
                    for row in r:
                        seq = list(tk.tokenize(row['body']))
                        # Skip bad data
                        if data_type == 'spoken_to':
                            # A tiny number of data are not formatted correctly
                            name = '@' + row[login].split('-')[0]
                            # Replace user name by <<MENTION>>
                            if name not in seq: 
                                continue
                            else:
                                pos = seq.index(name)
                                seq[pos] = '<<MENTION>>'
                        # save data to dictionary
                        tk_body = ' '.join(seq)
                        row['tk_body'] = tk_body
                        row['top_n_class'] = committer_dict[project][row[login]]['top_n_class']
                        row['top_n_rank'] = committer_dict[project][row[login]]['top_n_rank']
                        row['top_n_rank_cutoff'] = committer_dict[project][row[login]]['top_n_rank_cutoff']
                        row['top_n_metric_value'] = committer_dict[project][row[login]]['top_n_metric_value']
                        row['project'] = project
                        row['is_committer'] = row[login] in developerlist
                        row.pop('body')
                        rows.append(row)

                comments_dict[project] = rows

            pickle.dump(comments_dict, pkl_f)

    return comments_dict

def Using_one_project(project_idx, comments_dict):
    new_dict = defaultdict(dict)
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    project_name = projects[project_idx]

    for ind, project in enumerate(projects):
        # Only use the specified project
        if ind == project_idx:
            print('Using one project: ' + project)
            comment_list = comments_dict[project]
            new_dict[project] = comment_list

    return new_dict,chosen_projectname

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

# Get set of words to use.
def get_set(wordfile):
    s = set()

    with open(wordfile, "r") as myfile:
        for line in myfile.readlines():
            s.add(line.split()[0])  

    return s

# Only use a set of words. Replace other words by <<word>>
def set_of_words_only(comments_dict,wordfile):
    new_dict = defaultdict(dict)
    ccw = get_set(wordfile)

    for project in comments_dict:
        comment_list = comments_dict[project]
        for ind,d in enumerate(comment_list):
            seq = []
            for token in d['tk_body'].split(' '):
                if token in ccw:
                    seq.append(token)
                else: # use <ocw> to replace open class words
                    seq.append('<<word>>')
            comment_list[ind]['tk_body'] = ' '.join(seq)
        new_dict[project] = comment_list

    return new_dict

# Remove a set of words. Replace those words by <word>
def remove_set_of_words(comments_dict,wordfile):
    new_dict = defaultdict(dict)
    ccw = get_set(wordfile)

    for project in comments_dict:
        comment_list = comments_dict[project]
        for ind,d in enumerate(comment_list):
            seq = []
            for token in d['tk_body'].split(' '):
                if token in ccw:
                    seq.append('<<word>>')
                else:
                    seq.append(token)
            comment_list[ind]['tk_body'] = ' '.join(seq)
        new_dict[project] = comment_list

    return new_dict

# Returns train and test split idxs
def train_test_split_idxs(comments_dict, test_project, train_perc=0.7, seed=1337):
    tk_bodies = []
    random.seed(seed)
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    
    if test_project == -1: # no cross project validation
        test_project_name = "N.A."
        for project in projects:
            comment_list = comments_dict[project]
            for d in comment_list:
                tk_bodies.append(d['tk_body'])
        idxs = list(range(len(tk_bodies)))
        random.shuffle(idxs)
        train_idxs = idxs[ : int(train_perc * len(idxs))]
        test_idxs = idxs[int(train_perc * len(idxs)) : ]
    else: # cross project validation: test project #test_project
        test_project_name = projects[test_project]
        print('Testing on project: ' + projects[test_project])
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

    return train_idxs, test_idxs,test_project_name

def get_index_tokenizer(comments_dict, train_idxs, vocab_size=None):
    tk_bodies = []
    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    for project in projects:
        comment_list = comments_dict[project]
        for d in comment_list:
            tk_bodies.append(d['tk_body'])

    tkr = Tokenizer(num_words=vocab_size, filters='', lower=False, split=' ')
    tkr.fit_on_texts([tk_bodies[i] for i in train_idxs])

    return tkr

def remove_non_committer(comments_dict, committer_dict, data_type):
    clean_dict = defaultdict(dict)
    login = 'login' if data_type == 'speaking' else 'mention_login'

    for project in comments_dict:
        comment_list = comments_dict[project]
        clean_comments = []
        for d in comment_list:
            if d["is_committer"]:
                clean_comments.append(d)
        clean_dict[project] = clean_comments

    return clean_dict

def build_data(comments_dict, committer_dict, tokenizer, model_type, data_type):
    X, y, ids = [], [], []
    login = 'login' if data_type == 'speaking' else 'mention_login'

    # Need to guarantee an ordering
    projects = sorted(comments_dict.keys())
    for project in projects:
        comment_list = comments_dict[project]
        for d in comment_list:
            X.append(d['tk_body'])
            ids.append(d[login])
            if model_type == 'classifier':
                yval = 1 if committer_dict[project][d[login]]['top_n_class'] else 0
            else:
                yval = committer_dict[project][d[login]][model_type]
            y.append(yval)

    return X, y, ids

# we want to keep tokens around the <mention> when trimming sentence
def mention_preprocess(sequences, ids, max_len):
    newseqs = []

    for seq in sequences:
        seq = seq.split(' ');
        try:
            pos_at = seq.index('<<MENTION>>')
            if pos_at > max_len / 2: 
                seq = seq[pos_at - int(max_len / 2) :]
        except Exception as e:
            print('Warning: <<MENTION>> not found when doing mention preprocessing.')
        seq = ' '.join(seq)
        newseqs.append(seq)

    return newseqs

def trim_pad_sequences(sequences, max_len, pad_idx):
    newseqs = []

    for seq in sequences:
        if len(seq) > max_len:
            seq = seq[ : max_len]
        elif len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        newseqs.append(seq)

    return newseqs

def flatten_comments_dict(comments_dict):
    flattened = []

    projects = sorted(comments_dict.keys())
    for project in projects:
        comment_list = comments_dict[project]
        for d in comment_list:
            flattened.append(d)

    return flattened

def length_sequences(sequences):
    newseqs = []
    for seq in sequences:
        length = len(seq)
        newseqs.append([0]*length)

    return newseqs

def random_word(length):
    word = []
    for _ in range(length):
        word.append(random.choice(string.ascii_letters))

    return ''.join(word)

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