import pandas as pd
import numpy as np
import math
import argparse
import random
from collections import Counter
import os
import re
import json
import logging
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer



# --------------------- 数据读取 ---------------------
# 'user_id', 'item_id', 'ratings', 'date', 'reviews'


def get_count(tp, x_id):
    playcount_groupbyid = tp[[x_id, 'ratings']].groupby(x_id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def numerize(tp, user2id, item2id):
    uid = [user2id[x] for x in tp['user_id']]
    sid = [item2id[x] for x in tp['item_id']]
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp


def renumerize_uid(tp, user2id):
    uid = [user2id[x] for x in tp['user_id']]
    tp['user_id'] = uid
    return tp


def renumerize_iid(tp, item2id):
    sid = [item2id[x] for x in tp['item_id']]
    tp['item_id'] = sid
    return tp


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)  # 正则表达式替换
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(text, review_num, review_len, padding_word="<PAD/>"):
    new_text = {}
    for i in list(text.keys()):
        reviews = text[i]
        padded_reviews = []
        if len(reviews) > review_num:
            reviews = np.array(reviews)[np.random.choice(len(reviews), size=review_num, replace=False)].tolist()
        for ri in range(review_num):
            if ri < len(reviews):
                sentence = reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_reviews.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_reviews.append(new_sentence)
            else:
                new_sentence = [padding_word] * review_len
                padded_reviews.append(new_sentence)
        new_text[i] = padded_reviews
    return new_text


def load_json_file(file_path):
    users_id = []
    items_id = []
    ratings = []
    date = []
    reviews = []

    with open(file_path, 'rb') as f:
        for line in f:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown")
                continue
            if str(js['asin']) == 'unknown':
                print("unknown2")
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']) + ',')
            items_id.append(str(js['asin']) + ',')
            ratings.append(str(js['overall']))
            date.append(str(js['unixReviewTime']))
        f.close()

    print('Load done!')
    data = pd.DataFrame({'user_id': pd.Series(users_id),
                         'item_id': pd.Series(items_id),
                         'ratings': pd.Series(ratings),
                         'date': pd.Series(date),
                         'reviews': pd.Series(reviews)})[['user_id', 'item_id', 'ratings', 'date', 'reviews']]

    print('User num:{}'.format(data.user_id.nunique()))
    print('Item num:{}'.format(data.item_id.nunique()))
    print('Rating num:{}'.format(len(data)))

    return data


def load_review(df_data):
    user_reviews = {}
    item_reviews = {}
    user_rid = {}
    item_rid = {}
    for i in df_data.values:
        if i[0] in user_reviews:
            user_reviews[i[0]].append(i[4])
            user_rid[i[0]].append(i[1])
        else:
            user_rid[i[0]] = [i[1]]
            user_reviews[i[0]] = [i[4]]
        if i[1] in item_reviews:  #
            item_reviews[i[1]].append(i[4])
            item_rid[i[1]].append(i[0])
        else:
            item_reviews[i[1]] = [i[4]]
            item_rid[i[1]] = [i[0]]
    print('Load reviews done!')
    return user_reviews, item_reviews, user_rid, item_rid


def remove_stop_word(input_review, num_word):
    review = []
    idx = 0
    for user in input_review.keys():
        review.append('')
        for r in input_review[user]:
            review[idx] = review[idx] + ' ' + r.strip()
        review[idx] = clean_str(review[idx])
        idx = idx + 1
    vectorizer = TfidfVectorizer(max_df=0.5, stop_words={'english'}, max_features=num_word)
    vectorizer.fit(review)
    vocab = vectorizer.vocabulary_

    text = {}
    for i in input_review.keys():
        reviews = input_review[i]
        text[i] = []
        for s in reviews:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            s1_words = []
            for word in s1:
                if word in vocab.keys():
                    s1_words.append(word)
            text[i].append(s1_words)

    return text, list(vocab.keys())


def pro_review_with_stop_word(input_user_review, input_item_review, num_word):
    if 0 == num_word:
        num_word = None
    u_text, user_vocb = remove_stop_word(input_user_review, num_word)
    i_text, item_vocb = remove_stop_word(input_item_review, num_word)
    review_len_u = np.array([len(j) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)
    num_word_u = x2[int(0.9 * len(review_len_u)) - 1]
    num_word_u = 100
    print('num_word_u:{}'.format(num_word_u))
    review_len_i = np.array([len(j) for i in i_text.values() for j in i])
    y2 = np.sort(review_len_i)
    num_word_i = y2[int(0.9 * len(review_len_i)) - 1]

    num_word_i = 100
    print('num_word_i:{}'.format(num_word_i))
    print('Process reviews done!')

    return u_text, num_word_u, i_text, num_word_i


def pro_review(user_reviews, item_reviews, user_rid, item_rid):
    u_text = {}
    i_text = {}
    for i in list(user_reviews.keys()):
        u_reviews = user_reviews[i]
        u_text[i] = []
        for s in u_reviews:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            u_text[i].append(s1)
    for j in list(item_reviews.keys()):
        i_reviews = item_reviews[j]
        i_text[j] = []
        for s in i_reviews:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            i_text[j].append(s1)

    review_len_u = np.array([len(j) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)
    num_word_u = x2[int(0.5 * len(review_len_u)) - 1]
    print('num_word_u:{}'.format(num_word_u))
    review_len_i = np.array([len(j) for i in i_text.values() for j in i])
    y2 = np.sort(review_len_i)
    num_word_i = y2[int(0.5 * len(review_len_i)) - 1]
    print('num_word_i:{}'.format(num_word_i))
    print('Process reviews done!')
    return u_text, num_word_u, i_text, num_word_i


def build_vocab(sentences1, sentences2):
    word_counts = Counter(itertools.chain(*sentences1, *sentences2))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return vocabulary, vocabulary_inv


def build_input_data(u_text, i_text, vocabulary):
    u_text2 = {}
    for i in list(u_text.keys()):
        u_reviews = u_text[i]
        u = np.array([[vocabulary[word] for word in words] for words in u_reviews])
        u_text2[i] = u

    i_text2 = {}
    for j in list(i_text.keys()):
        i_reviews = i_text[j]
        i = np.array([[vocabulary[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def get_review_num(user_reviews, item_reviews):
    review_num_u = np.array([len(x) for x in user_reviews.values()])
    x = np.sort(review_num_u)
    u_review_num = x[int(0.5 * len(review_num_u)) - 1]
    review_num_u_count = Counter(x)

    review_num_i = np.array([len(x) for x in item_reviews.values()])
    y = np.sort(review_num_i)
    i_review_num = y[int(0.5 * len(review_num_i)) - 1]
    review_num__count = Counter(y)
    return u_review_num, i_review_num


def select_reviews(text, review_num):
    for i in list(text.keys()):
        reviews = text[i]
        if len(reviews) > review_num:
            reviews = np.array(reviews)[np.random.choice(len(reviews), size=review_num, replace=False)].tolist()
            text[i] = reviews
    return text


def get_reviews(u_file, i_file, vocabulary_old):
    user_reviews_old = {}
    item_reviews_old = {}
    vocabulary_reverse = {value: key for key, value in vocabulary_old.items()}
    for u_id in list(u_file.keys()):
        u_reviews = u_file[u_id]
        u_reviews_new = [[vocabulary_reverse[w_id] for w_id in u_re] for u_re in u_reviews]
        user_reviews_old[u_id] = u_reviews_new
    for i_id in list(i_file.keys()):
        i_reviews = i_file[i_id]
        i_reviews_new = [[vocabulary_reverse[w_id] for w_id in i_re] for i_re in i_reviews]
        item_reviews_old[i_id] = i_reviews_new
    print('get_reviews done!')
    return user_reviews_old, item_reviews_old


def pad_reviews(reviews_words, review_num, num_word=10, padding_word="<PAD/>"):
    new_text = {}
    for i in list(reviews_words.keys()):
        reviews = reviews_words[i]
        padded_reviews = []
        for ri in range(review_num):
            if ri < len(reviews):
                cur_review = reviews[ri]
                if num_word > len(cur_review):
                    num_padding = num_word - len(cur_review)
                    padded_review = cur_review + [padding_word] * num_padding
                    padded_reviews.append(padded_review)
                else:
                    padded_reviews.append(cur_review)
            else:
                padded_review = [padding_word] * num_word
                padded_reviews.append(padded_review)
        new_text[i] = padded_reviews
    return new_text


def pro_data(data_origin):
    usercount = get_count(data_origin, 'user_id')
    itemcount = get_count(data_origin, 'item_id')
    unique_uid = usercount.user_id
    unique_sid = itemcount.item_id
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    data_numerize = numerize(data_origin, user2id, item2id)
    print('ID numerize done!')
    return data_numerize


def split_train_test(df_data, partition_rate=0.7, threshold=10):
    df_data[['user_id', 'item_id', 'date']] = df_data[['user_id', 'item_id', 'date']].astype(np.int32)
    df_data[['ratings']] = df_data[['ratings']].astype(np.float32)

    df_data.sort_values(by=['user_id', 'date'], inplace=True)
    df_data = df_data[df_data['user_id'].groupby(df_data['user_id']).transform('size') >= threshold]

    usercount = get_count(df_data, 'user_id')
    unique_uid = usercount.user_id
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    df_data = renumerize_uid(df_data, user2id)

    itemcount = get_count(df_data, 'item_id')
    unique_sid = itemcount.item_id
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    df_data = renumerize_iid(df_data, item2id)

    print('User num:{}'.format(df_data.user_id.nunique()))
    print('Item num:{}'.format(df_data.item_id.nunique()))
    print('Rating num:{}'.format(len(df_data)))
    print('ID renumerize done!')

    user_reviews_old, item_reviews_old = get_reviews(u_file, i_file, vocabulary_old)

    user_reviews, item_reviews, user_rid, item_rid = load_review(df_data)
    u_review_num, i_review_num = get_review_num(user_reviews, item_reviews)
    print('u_review_num:{}'.format(u_review_num))
    print('i_review_num:{}'.format(i_review_num))
    user_reviews = select_reviews(user_reviews, u_review_num)
    item_reviews = select_reviews(item_reviews, i_review_num)

    u_text_, num_word_u, i_text_, num_word_i = pro_review_with_stop_word(user_reviews, item_reviews, 35000)
    u_text = pad_sentences(u_text_, u_review_num, num_word_u)
    print("pad user done")
    i_text = pad_sentences(i_text_, i_review_num, num_word_i)
    print("pad item done")
    user_voc = [xx for x in u_text.values() for xx in x]
    item_voc = [xx for x in i_text.values() for xx in x]
    vocabulary, vocabulary_inv = build_vocab(user_voc, item_voc)
    u_text, i_text = build_input_data(u_text, i_text, vocabulary)

    if not os.path.exists(opt.dataset):
        os.makedirs(opt.dataset)
    pickle.dump(u_text, open(os.path.join(OUT_DIR, 'u_text_sw_100'), 'wb'))
    pickle.dump(i_text, open(os.path.join(OUT_DIR, 'i_text_sw_100'), 'wb'))
    pickle.dump(vocabulary, open(os.path.join(OUT_DIR, 'vocabulary_sw_100'), 'wb'))
    print('review files and vocabulary done!')

    df_data = df_data[['user_id', 'item_id', 'ratings', 'date']]

    unique_uids = df_data['user_id'].unique()
    df_user_interactions = []
    for uuid in unique_uids:
        df_user_interaction = df_data[df_data['user_id'].isin([uuid])]
        df_user_interactions.append(df_user_interaction)

    df_train = []
    df_test = []
    for df_user in df_user_interactions:
        df_utrain = df_user[:math.floor(len(df_user)*partition_rate)]
        df_utest = df_user[math.floor(len(df_user)*partition_rate):]
        df_train.append(df_utrain)
        df_test.append(df_utest)
    df_data_train = pd.concat(df_train)
    df_data_test = pd.concat(df_test)

    print('All done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Amazon_Instant_Video', help='Amazon_Instant_Video/Books/Beauty/Pet_Supplies/Tools_and_Home_Improvement')
    parser.add_argument('--cuda', type=str, default='2', help='which gpu to run')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)

    np.random.seed(2022)
    IN_DIR = '../../data/Amazon/'
    json_name = 'reviews_'+opt.dataset+'.json'
    OUT_DIR = '../datasets/' + opt.dataset + '/'
    print('dataset:' + opt.dataset)
    f1 = open('../datasets/Instant_Video/u_text', 'rb')
    f2 = open('../datasets/Instant_Video/i_text', 'rb')
    f3 = open('../datasets/Instant_Video/vocabulary', 'rb')
    u_file = pickle.load(f1)
    u_text_old = np.array([uu.flatten() for uu in u_file.values()])
    i_file = pickle.load(f2)
    i_text_old = np.array([ii.flatten() for ii in i_file.values()])
    vocabulary_old = pickle.load(f3)

    data = load_json_file(os.path.join(IN_DIR, json_name))
    data = pro_data(data)
    split_train_test(data, threshold=10)