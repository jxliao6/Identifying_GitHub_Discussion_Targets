import pickle
import numpy as np
import sys
import array
import os
from joblib import Parallel, delayed
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity



def create_indexed_dataset(pklpath_reply, pklpath_nonreply):
    rank = build_vocabulary_from_pkls(pklpath_reply, pklpath_nonreply)

    with open(pklpath_reply, 'rb') as pkl_f:
        reply = pickle.load(pkl_f)
    with open(pklpath_nonreply, 'rb') as pkl_f:
        nonreply = pickle.load(pkl_f)
    # Now convert bodies into
    # integers based on the frequency
    indexed_results_reply = []
    for result in reply:
        # Need to copy to avoid overwriting the function argument
        r = result.copy()
        r['lemmed_body'] = [rank[token] for token in result['lemmed_body']]
        indexed_results_reply.append(r)
    indexed_results_nonreply = []
    for result in nonreply:
        # Need to copy to avoid overwriting the function argument
        r = result.copy()
        r['lemmed_body'] = [rank[token] for token in result['lemmed_body']]
        indexed_results_nonreply.append(r)

    return (indexed_results_reply, indexed_results_nonreply)

def read_pkl_data_list(pkl_fps):
    data_full = []
    for fp in pkl_fps:
        with open(fp, 'rb') as pkl_f:
            d = pickle.load(pkl_f)
            data_full += d
    return data_full

# Creates a COMBINED vocabulary from both reply and nonreply data.
# This way we have a coherent indexing scheme for reply and nonreply data,
# as is necessary for our methods to work properly.
def build_vocabulary_from_pkls(pklpath_reply, pklpath_nonreply):
    with open(pklpath_reply, 'rb') as pkl_f:
        reply = pickle.load(pkl_f)
    with open(pklpath_nonreply, 'rb') as pkl_f:
        nonreply = pickle.load(pkl_f)

    all_words = []
    for r in reply:
        all_words += r['lemmed_body']
    for nr in nonreply:
        all_words += nr['lemmed_body']

    freqs = Counter(all_words).most_common()
    # rank dict is our vocabulary
    rank = {pair[0]: rank + 1 for rank, pair in enumerate(freqs)}

    return rank

# Assumes embedding is a numpy array with shape (vocab_size, embedding_length)
# where the row number corresponds to the word index.
# Calculates cosine similarity
def calc_closest_embedding(idx, idx2word, embedding, top_words=5):
    emb = embedding[idx, :].reshape(1, -1)
    cosines = {}
    for i in range(embedding.shape[0]):
        cosines[i] = cosine_similarity(emb, embedding[i, :].reshape(1, -1))[0, 0]
    sorted_cosines = sorted(cosines.items(), key=lambda x: x[1], reverse=True)
    # Offset by 1 since the closest thing to the word is the word itself
    return [(idx2word[i], v) for (i, v) in sorted_cosines[1:top_words + 1]]

# def calc_sim(i, j, embedding):
#     return (i, j, cosine_similarity(embedding[i, :].reshape(1, -1),
#                                     embedding[j, :].reshape(1, -1)))

# Wrapper for use with top percentages
def _calc_top1_embedding_widx(idx, idx2word, embedding, top_words):
    result = calc_closest_embedding(idx, idx2word, embedding, top_words)
    return (idx2word[idx], result[0])

def calc_top_perc_similarity(idx2word, embedding, perc, filter_idxs=None):
    idxs = idx2word.keys() if filter_idxs is None else filter_idxs
    # Map of word : (closest word, similarity)
    closest = {}
    results = Parallel(n_jobs=16, verbose=5)(delayed(_calc_top1_embedding_widx)(idx, idx2word, embedding, 1) \
                                                     for idx in idxs)
    for result in results:
        closest[result[0]] = result[1]
    sorted_closest = sorted(closest.items(), key=lambda x: x[1][1], reverse=True)
    sorted_closest_perc = sorted_closest[:int(perc * len(sorted_closest))]
    
    ret_map = {}
    for item in sorted_closest_perc:
        ret_map[item[0]] = item[1]
    return ret_map

def load_cached_data(hashh):
    hash_f = get_cache_path(hashh)
    if os.path.exists(hash_f):
        with open(hash_f, 'rb') as pkl_f:
            print('Loading cached data %s' % hash_f)
            return pickle.load(pkl_f)
    else:
        return None

def get_cache_path(hashh, prefix='./'):
    return os.path.join(prefix + 'data_cache', hashh.hexdigest())

# "Default" values of readability scores are based on
# the particular score. If there are no sentences or no words,
# return the "hardest" value (these things aren't bounded,
# but the values returned are based on specified ranges for interpretation)
def flesch_reading_ease(word_count, sent_count, syllable_count):
    if sent_count == 0 or word_count == 0:
        return 0
    return 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syllable_count / word_count)


def flesch_kincaid_grade_level(word_count, sent_count, syllable_count):
    if sent_count == 0 or word_count == 0:
        return 12
    return 0.39 * (word_count / sent_count) + 11.8 * (syllable_count / word_count) - 15.9


def automated_reading_index(word_count, character_count, sent_count):
    if sent_count == 0 or word_count == 0:
        return 14
    return 4.71 * (character_count / word_count) + 0.5 * (word_count / sent_count) - 21.43




# def calc_pairwise_similarities(idx2word, embedding):
#     # Similarity matrix
#     # TODO: can make this way better since it's symmetric
#     # but I can't figure out the index translation right now
#     sims = []
#     for _ in range(embedding.shape[0]):
#         sims.append(array.array('f', (0,)*embedding.shape[0]))
#     idxs = idx2word.keys()
#     results = Parallel(n_jobs=16, verbose=5)(delayed(calc_sim)(i, j, embedding) \
#                                             for i in range(len(idxs)) \
#                                                 for j in range(len(idxs)))
#     pickle.dump(results, open('temp.pkl', 'wb'))
#     print('Constructing similarity matrix')
#     for i in range(len(idxs)):
#         for j in range(len(idxs)):
#             sims[i][j] = results[i * len(idxs) + j]
#     print('Done constructing similarity matrix')

#     # print(len(results))
#     # prev_offset = 0
#     # for i in range(len(idxs)):
#     #     base_index = i * len(idxs) + j
#     #     prev_offset = prev_offset + i
#     #     for j in range(i, len(idxs)):
#     #         print((i, j))
#     #         print(i * len(idxs) + j)
#     #         sims[i][j] = results[base_index - (prev_offset + i)]
#     return sims
