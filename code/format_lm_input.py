import argparse, os, csv, sys, hashlib, pickle, random, string
csv.field_size_limit(sys.maxsize)

from nltk.tokenize.casual import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from util import load_cached_data, get_cache_path

COMMIT_METRICS = ['additions', 'deletions', 'total', 'commits']

# Form of committer data dict:
# d[project][author (login)] = dict of key value pairs for relevant columns e.g.
#                    total number of commits, total number of additions, etc
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
      author_data['top_n_class'] = authors.index(author) >= len(metrics) - \
        top_perc / 100 * len(metrics)
      author_data['top_n_rank'] = authors.index(author)
      author_data['top_n_rank_cutoff'] = len(metrics) - top_perc / 100 * \
        len(metrics)
      author_data['top_n_metric_value'] = metrics[authors.index(author)]

    data_dict[project] = author_data_dict

  return data_dict

# Form of comments_dict
# d[project] = list of dict of key value pairs for each comment
def create_comments_dict(base_dir, committer_dict, comments_file, model_type, hashh=None):
  comments_dict = None
  login = 'login' if model_type == 'speaking' else 'mention_login'

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
        rows = []
        with open(os.path.join(root, comments_file), 'r') as inf:
          r = csv.DictReader(inf)
          for row in r:
            seq = list(tk.tokenize(row['body']))
            # skip bad data
            if model_type == 'spoken_to':
              name = '@' + row[login]
              if name not in seq: 
                continue
              else:
                pos = seq.index(name)
                seq[pos] = '<<MENTION>>'
            # save data to dictionary
            tk_body = ' '.join(seq)
            row['tk_body'] = tk_body
            row['top_n_class'] = \
              committer_dict[project][row[login]]['top_n_class']
            row['top_n_rank'] = \
              committer_dict[project][row[login]]['top_n_rank']
            row['top_n_rank_cutoff'] = \
              committer_dict[project][row[login]]['top_n_rank_cutoff']
            row['top_n_metric_value'] = \
              committer_dict[project][row[login]]['top_n_metric_value']
            row['project'] = project
            row.pop('body')
            rows.append(row)
        comments_dict[project] = rows

      pickle.dump(comments_dict, pkl_f)

  return comments_dict

def Using_one_project(project_num, comments_dict):
  new_dict = defaultdict(dict)
  # Need to guarantee an ordering
  projects = sorted(comments_dict.keys())

  for ind, project in enumerate(projects):
    if ind == project_num:
      with open('lm_result','a') as file:
        file.write('Using project: ' + project + '\n')
      comment_list = comments_dict[project]
      new_dict[project] = comment_list

  return new_dict

def remove_short_comments(comments_dict, noise_len):
  clean_dict = defaultdict(dict)

  for project in comments_dict:
    comment_list = comments_dict[project]
    clean_comments = []
    for d in comment_list:
      if len(d['tk_body'].split()) > noise_len:
        clean_comments.append(d)
    clean_dict[project] = clean_comments

  return clean_dict

def get_ccw_set():
  s = {'<S_CODE_TOKEN>'}

  with open("closed_class_words", "r") as myfile:
    for line in myfile.readlines():
      s.add(line.split()[0])  

  return s

def closed_class_only(comments_dict):
  new_dict = defaultdict(dict)
  ccw = get_ccw_set()

  for project in comments_dict:
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

def remove_closed_class(comments_dict):
  new_dict = defaultdict(dict)
  ccw = get_ccw_set()

  for project in comments_dict:
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

# Returns train and test split idxs
def train_test_split(comments_dict, test_project, train_perc=0.7, seed=1337):
  tk_bodies = []
  random.seed(seed)
  # Need to guarantee an ordering
  projects = sorted(comments_dict.keys())
  
  if test_project == -1: # no cross project validation
    for project in projects:
      comment_list = comments_dict[project]
      for d in comment_list:
        tk_bodies.append(d['tk_body'])
    idxs = list(range(len(tk_bodies)))
    random.shuffle(idxs)
    train_idxs = idxs[ : int(train_perc * len(idxs))]
    test_idxs = idxs[int(train_perc * len(idxs)) : ]
  else: 
    train_idxs = []
    test_idxs = []
    index = 0
    for ind, project in enumerate(projects):
      comment_list = comments_dict[project]
      if ind == test_project:
        with open('lm_result','a') as file:
          file.write('testing project: ' + project + '\n')
      for d in comment_list:
        if ind == test_project:
          test_idxs.append(index)
        else:
          train_idxs.append(index)
        index += 1
  return train_idxs, test_idxs

def get_index_tokenizer(comments_dict, train_idxs, num_words):
  tk_bodies = []
  # Need to guarantee an ordering
  projects = sorted(comments_dict.keys())
  for project in projects:
    comment_list = comments_dict[project]
    for d in comment_list:
      tk_bodies.append(d['tk_body'])

  tkr = Tokenizer(num_words=num_words, filters='', lower=False, split=' ')
  tkr.fit_on_texts([tk_bodies[i] for i in train_idxs])

  return tkr

def remove_non_committer(comments_dict, committer_dict, model_type):
  clean_dict = defaultdict(dict)
  login = 'login' if model_type == 'speaking' else 'mention_login'

  for project in comments_dict:
    comment_list = comments_dict[project]
    clean_comments = []
    for d in comment_list:
      if committer_dict[project][d[login]]["top_n_metric_value"]==0 :
        continue
      clean_comments.append(d)
    clean_dict[project] = clean_comments

  return clean_dict

def build_data(comments_dict, committer_dict, tokenizer, model_type):
  X, y, ids = [], [], []
  login = 'login' if model_type == 'speaking' else 'mention_login'

  # Need to guarantee an ordering
  projects = sorted(comments_dict.keys())
  for project in projects:
    comment_list = comments_dict[project]
    for d in comment_list:
      X.append(d['tk_body'])
      ids.append(d[login])
      yval = 1 if committer_dict[project][d[login]]['top_n_class'] else 0
      y.append(yval)

  return X, y, ids

# we want to keep tokens around the <mention> when trimming sentence
def mention_preprocess(sequences, ids, max_len):
  newseqs = []

  for seq in sequences:
    seq = seq.split(' ');
    try:
      pos_at = seq.index('<mention>')
      if pos_at > max_len / 2: 
        seq = seq[pos_at - int(max_len / 2) :]
    except Exception as e:
      print('<mention> not_found!')
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
