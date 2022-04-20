import os
from collections import Counter
from pytrends.request import TrendReq
import nltk
from nltk.corpus import stopwords, words
from nltk.stem.porter import PorterStemmer
import json
import gzip
import pickle
import time
from config import PRE_DATASET_FOLDER, RAW_DATASET_FOLDER, PRO_DATASET_FOLDER
import numpy as np
import tarfile
import email
import mailbox
import csv
import pandas as pd
import re
import gtab


def find_most_popular_keyword(keywords, nweeks):
    pytrend = TrendReq()

    most_popular = keywords[0]
    for i in range(1, len(keywords), 4):
        time.sleep(5)  # This ensures that 20 loops (querying for 100 words) takes at least 100 seconds.
        if i + 4 <= len(keywords):
            keywords_to_search = keywords[i:i + 4]
        else:
            keywords_to_search = keywords[i:]
        keywords_to_search += [most_popular]
        nread = len(keywords_to_search)
        print("({:d}/{:d}) Searching for:".format(i, len(keywords)), keywords_to_search, flush=True)

        pytrend.build_payload(kw_list=keywords_to_search)
        pytrend.build_payload(kw_list=keywords_to_search, geo='US', timeframe='2020-01-01 2020-12-31')
        data = pytrend.interest_over_time().to_numpy()

        mini_matrix = np.zeros((nread, nweeks))
        for i_kw in range(nread):
            for j, i_week in enumerate(range(-nweeks - 1, -1)):
                mini_matrix[i_kw, j] = data[i_week][i_kw]
        # print(mini_matrix)

        idx_max = np.argmax(np.max(mini_matrix, axis=1))
        if idx_max < nread - 1:
            print("{:s} replaced {:s} as most popular!".format(keywords_to_search[idx_max], most_popular), flush=True)
            most_popular = keywords_to_search[idx_max]

    return most_popular


def get_keyword_trends(keywords, baseline_keyword, nweeks):
    pytrend = TrendReq()
    nkw = len(keywords)

    trend_matrix = np.zeros((nkw, nweeks))
    for i in range(0, nkw, 4):
        time.sleep(5)  # This ensures that 20 loops (querying for 100 words) takes at least 100 seconds.
        if i + 4 <= nkw:
            keywords_to_search = keywords[i:i + 4]
        else:
            keywords_to_search = keywords[i:]
        nread = len(keywords_to_search)
        print(keywords_to_search)

        # Get mini_matrix and norm_factor, special case if the baseline word is in the kw to search
        if baseline_keyword in keywords_to_search:
            i_baseline = keywords_to_search.index(baseline_keyword)
            pytrend.build_payload(kw_list=keywords_to_search)
            data = pytrend.interest_over_time().to_numpy()

            mini_matrix = np.zeros((nread, nweeks))
            for i_kw in range(nread):
                for j, i_week in enumerate(range(-nweeks - 1, -1)):
                    mini_matrix[i_kw, j] = data[i_week][i_kw]

            norm_factor = np.sum(mini_matrix[i_baseline])
            if norm_factor == 0:
                norm_factor = 1

        else:
            pytrend.build_payload(kw_list=keywords_to_search + [baseline_keyword])
            data = pytrend.interest_over_time().to_numpy()

            mini_matrix = np.zeros((nread, nweeks))
            for i_kw in range(nread):
                for j, i_week in enumerate(range(-nweeks - 1, -1)):
                    mini_matrix[i_kw, j] = data[i_week][i_kw]

            baseline_trend = np.zeros(nweeks)
            for j, i_week in enumerate(range(-nweeks - 1, -1)):
                baseline_trend[j] = data[i_week][nread]
            norm_factor = np.sum(baseline_trend)
            if norm_factor == 0:
                norm_factor = 1

        print(mini_matrix)

        trend_matrix[i:i + nread] = mini_matrix / norm_factor
        for k in range(0, nread):
            print("{:s} ({:d}/{:d}) :".format(keywords[i + k], i + k, nkw), end='')
            print(trend_matrix[i + k][-5:], end='')
            print("    sum={:.2f}".format(sum(trend_matrix[i + k])), flush=True)

    return trend_matrix


def dataset_of_words_to_ids(dataset):
    unique_keywords = np.unique([kw for document in dataset for kw in document])
    kw_to_id = {kw: i for i, kw in enumerate(unique_keywords)}
    dataset = [[kw_to_id[kw] for kw in document] for document in dataset]
    return dataset, unique_keywords


def extract_words_from_original_dataset(dataset_original):
    """Receives an original dataset: a list of strings, where each string is a document.
    Extracts the words from the document using a regular expression, converting to lower case, and keeping only unique alpha-only words"""
    dataset_keywords = []
    for document in dataset_original:
        unique_words_this_doc = list(set(re.findall(r'\w+', document)))
        unique_words_this_doc = list(set([word.lower() for word in unique_words_this_doc if word.isalpha()]))
        if len(unique_words_this_doc) > 0:
            dataset_keywords.append(unique_words_this_doc)
    return dataset_keywords


def process_email(message):
    payload = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            payload.append(part.get_payload())
    payload = "".join(payload)
    return payload


def preprocess_raw_dataset(dataset_name, force_recompute=False):
    # Dataset names: 'enron-full', 'enron-sent', 'lucene', 'bow-enron', 'bow-kos', ...

    path_to_pre_dataset = os.path.join(PRE_DATASET_FOLDER, dataset_name + '.pkl')
    if os.path.exists(path_to_pre_dataset) and not force_recompute:
        print("Pre-processed dataset {:s} already existed, not processing again.".format(dataset_name))
        return

    time0 = time.time()
    print("Going to process {:s}".format(dataset_name))
    if dataset_name == 'enron-full':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'enron_mail_20150507.tar.gz')
        # Get the _sent_mail folder only:
        dataset_original = []
        count = 0
        with tarfile.open(path_to_raw_dataset, mode='r') as tar:
            for member in tar.getmembers():
                if '_sent_mail' in member.path:
                    if member.isfile():
                        file_object = tar.extractfile(member)
                        email_data_binary = file_object.read()
                        email_data_string = email_data_binary.decode("utf-8")
                        message = email.message_from_string(email_data_string)
                        dataset_original.append(process_email(message))
                        count += 1
                        if count % 1000 == 0:
                            print("{:d} documents processed".format(count))
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords = extract_words_from_original_dataset(dataset_original)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords), dataset_name))
        dataset, unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset, unique_keywords)
    elif dataset_name == 'lucene':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'lucene-java-user')
        dataset_original = []
        count = 0
        for filename in os.listdir(path_to_raw_dataset):
            if filename.endswith('mbox'):
                mbox = mailbox.mbox(os.path.join(path_to_raw_dataset, filename))
                for message in mbox:
                    dataset_original.append(process_email(message).split("To unsubscribe")[0])
                    count += 1
                    if count % 1000 == 0:
                        print("{:d} documents processed".format(count))
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords = extract_words_from_original_dataset(dataset_original)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords), dataset_name))
        dataset, unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset, unique_keywords)
    elif dataset_name.startswith('bow-'):
        name = dataset_name.split('-')[1]
        assert name in ('enron', 'kos', 'nips', 'nytimes', 'pubmed')
        path_to_raw_docword = os.path.join(RAW_DATASET_FOLDER, 'docword.{:s}.txt.gz'.format(name))
        path_to_raw_vocabulary = os.path.join(RAW_DATASET_FOLDER, 'vocab.{:s}.txt'.format(name))
        with open(path_to_raw_vocabulary, 'rt') as f:
            vocabulary = f.read().splitlines()
        with gzip.open(path_to_raw_docword, 'rt') as f:
            docword = f.read().splitlines()
        docword_matrix = np.array([line.split(' ') for line in docword[3:]], dtype=int)
        print("Done reading and turning docword into matrix ({:.0f} secs)".format(time.time() - time0))

        dataset = []
        doc_iterator = iter(docword_matrix[:, 0])
        idx_new = 0
        current_doc = 1
        for idx, doc_id in enumerate(doc_iterator):
            if doc_id > current_doc:
                current_doc = doc_id
                dataset.append(list(docword_matrix[idx_new: idx, 1]))
                idx_new = idx
        dataset.append(list(set(docword_matrix[idx_new:, 1])))

        num_rows = docword_matrix.shape[0]
        num_rows2 = sum([len(val) for val in dataset])
        print("Sanity check : {:d}={:d}".format(num_rows, num_rows2))
        assert num_rows == num_rows2

        results = (dataset, vocabulary)
    elif dataset_name == 'articles1':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'articles1.csv')
        df = pd.read_csv(path_to_raw_dataset)
        dataset_original = list(df['content'])
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords = extract_words_from_original_dataset(dataset_original)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords), dataset_name))
        dataset, unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset, unique_keywords)
    elif dataset_name == 'book-summaries':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'booksummaries/booksummaries.txt')
        with open(path_to_raw_dataset, 'rt') as f:
            summaries = f.readlines()
        dataset_original = [summary.split('\t')[-1] for summary in summaries]
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords = extract_words_from_original_dataset(dataset_original)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords), dataset_name))
        dataset, unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset, unique_keywords)
    elif dataset_name == 'movie-plots':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'MovieSummaries/plot_summaries.txt')
        with open(path_to_raw_dataset, 'rt') as f:
            summaries = f.readlines()
        dataset_original = [summary.split('\t')[-1] for summary in summaries]
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords = extract_words_from_original_dataset(dataset_original)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords), dataset_name))
        dataset, unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset, unique_keywords)
    else:
        raise ValueError("dataset_name {:s} is not ready".format(dataset_name))

    with open(path_to_pre_dataset, 'wb') as f:
        pickle.dump(results, f)
    print("Done pre-processing, saved {:s}, elapsed time ({:.0f} secs)".format(path_to_pre_dataset, time.time() - time0))


def process_pre_dataset(dataset_name, nkw=3000, force_recompute=False):
    path_to_pro_dataset = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
    if os.path.exists(path_to_pro_dataset) and not force_recompute:
        print("Processed dataset {:s} already existed, not processing again.".format(dataset_name))
        return

    time0 = time.time()
    print("Going to process {:s}".format(dataset_name))
    with open(os.path.join(PRE_DATASET_FOLDER, dataset_name + '.pkl'), 'rb') as f:
        dataset, vocabulary = pickle.load(f)

    # Remove stopwords, words with more than 20 characters, and less than 2 characters, and stem all the other words
    nltk.download('stopwords')
    english_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    stem_vocabulary = [ps.stem(word) if (word.lower() not in english_stopwords and 2 < len(word) < 20 and word.isalpha()) else None for word in vocabulary]
    stem_vocabulary_unique = sorted(list(set(stem_vocabulary) - {None}))
    kwid_to_stemid = {i: stem_vocabulary_unique.index(stem) for i, (word, stem) in enumerate(zip(vocabulary, stem_vocabulary)) if stem is not None}
    new_dataset = [sorted(list(set([kwid_to_stemid[kwid] for kwid in document if kwid in kwid_to_stemid]))) for document in dataset]
    print("Done stemming and trimming dataset, current size {:d} ({:.0f} secs)".format(len([id for document in new_dataset for id in document]), time.time() - time0))

    # Select top nkw step ID's
    stemid_counter = Counter([stemid for stemid_list in new_dataset for stemid in stemid_list])
    sorted_stemid = sorted(stemid_counter.keys(), key=lambda x: stemid_counter[x], reverse=True)
    chosen_stemid = sorted_stemid[:nkw]
    old_to_new_stemid = {old: new for new, old in enumerate(chosen_stemid)}
    new_dataset = [sorted(list(set([old_to_new_stemid[stemid] for stemid in document if stemid in old_to_new_stemid]))) for document in new_dataset]
    print("Done choosing top stem ID's and trimming dataset, current size {:d} ({:.0f} secs)".format(len([id for document in new_dataset for id in document]), time.time() - time0))

    print("Dataset after removing empty documents: {:d} -> ".format(len(new_dataset)), end="")
    new_dataset = [document for document in new_dataset if len(document) > 0]
    print("{:d} documents".format(len(new_dataset)))

    chosen_stems = [stem_vocabulary_unique[chosen_stemid[i]] for i in range(nkw)]
    stems_to_words = {stem: [] for stem in chosen_stems}
    for word, stem in zip(vocabulary, stem_vocabulary):
        if stem in stems_to_words:
            stems_to_words[stem].append(word)

    results = (new_dataset, chosen_stems, {'stems_to_words': stems_to_words})

    with open(path_to_pro_dataset, "wb") as f:
        pickle.dump(results, f)
    print("Written processed dataset to {}!".format(path_to_pro_dataset))

    return results


def get_frequencies_from_google_trends(dataset_list):
    """Oct 28th 2020.
    Gets query frequencies from google trends using gtab for the right normalization. It stores the results in a database."""

    def load_keyword_trends():
        path_to_keyword_trends = os.path.join(PRO_DATASET_FOLDER, 'trends.pkl')
        if os.path.exists(path_to_keyword_trends):
            with open(path_to_keyword_trends, "rb") as f:
                keyword_trends = pickle.load(f)
            print("Loaded keyword_trends, we have {:d} keywords".format(len(keyword_trends), flush=True))
        else:
            keyword_trends = {}
            print("Creating keyword_trends...", flush=True)
        return keyword_trends

    def save_keyword_trends(keyword_trends):
        path_to_keyword_trends = os.path.join(PRO_DATASET_FOLDER, 'trends.pkl')
        with open(path_to_keyword_trends, "wb") as f:
            pickle.dump(keyword_trends, f)

    keyword_trends = load_keyword_trends()
    new_keywords = []
    for dataset_name in dataset_list:
        path_to_pro_dataset = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
        with open(path_to_pro_dataset, "rb") as f:
            dataset, stems, aux = pickle.load(f)
        stems_to_words = aux['stems_to_words']
        keywords = [kw for kw_list in stems_to_words.values() for kw in kw_list]
        for kw in keywords:
            if kw not in keyword_trends and kw not in new_keywords:
                new_keywords.append(kw)
    print("Done scanning the datasets, we have {:d} new keywords".format(len(new_keywords)), flush=True)

    if len(new_keywords) > 0:
        t = gtab.GTAB()  # We use the default anchor
        # t.set_options(pytrends_config={'timeframe': '2020-01-01 2020-12-31'}, gtab_config={'sleep': 2})
        t.set_options(pytrends_config={'timeframe': '2020-01-01 2020-12-31'})
        for i, kw in enumerate(new_keywords):
            try:
                query = t.new_query(str(kw))
            except BaseException as e:
                print("There was an exception, so we are saving...")
                save_keyword_trends(keyword_trends)
                print("Saved!")
                raise
            keyword_trends[kw] = query
            # save_keyword_trends(keyword_trends)
            print("Added '{:s}', {:d} left".format(kw, len(new_keywords) - i - 1), flush=True)
        save_keyword_trends(keyword_trends)
        print("Done adding all keywords!")


def add_frequency_trends_information_to_dataset(dataset_name):
    path_to_pro_dataset = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
    with open(path_to_pro_dataset, "rb") as f:
        dataset, stems, aux = pickle.load(f)
    print("Loaded", dataset_name)

    path_to_keyword_trends = os.path.join(PRO_DATASET_FOLDER, 'trends.pkl')
    with open(path_to_keyword_trends, "rb") as f:
        keyword_trends = pickle.load(f)
    print("Loaded keyword_trends, we have {:d} keywords".format(len(keyword_trends), flush=True))

    words = [word for stem in stems for word in aux['stems_to_words'][stem]]
    if not all([word in keyword_trends for word in words]):
        print("Not all words in {:s} are in the trend dictionary, we skip this dataset".format(dataset_name))
    elif 'trends' in aux:
        print("aux['trends'] already existed, we skip this dataset")
    else:
        print("Computing trends matrix...")
        trends_matrix = np.zeros((len(stems), 52))
        for i_stem, stem in enumerate(stems):
            for word in aux['stems_to_words'][stem]:
                if isinstance(keyword_trends[word], pd.DataFrame):
                    trends_matrix[i_stem, :] += keyword_trends[word]['max_ratio'].values

        aux['trends'] = trends_matrix

        results = (dataset, stems, aux)

        with open(path_to_pro_dataset, "wb") as f:
            pickle.dump(results, f)
        print("Written dataset to {}!".format(path_to_pro_dataset))


if __name__ == "__main__":

    os.system('mesg n')
    dataset_list = ['enron-full', 'lucene', 'bow-nytimes', 'articles1', 'movie-plots']
    # get_frequencies_from_google_trends(dataset_list)
    for dataset_name in dataset_list:
        preprocess_raw_dataset(dataset_name)
        process_pre_dataset(dataset_name)
    # for dataset_name in dataset_list:
    #     add_frequency_trends_information_to_dataset(dataset_name)
