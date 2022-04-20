This repository contains the code to evaluate query recovery attacks in Searchable Symmetric Encryption (SSE) schemes.

This code can be used to reproduce the results in:
* Simon Oya and Florian Kerschbaum. *"IHOP: Improved Statistical Query Recovery against Searchable Symmetric Encryption through Quadratic Optimization."* (2022).

**DISCLAIMER**: the code should work, but there is a lot of room for improvements. There are many "TODO" comments here and there, but I might not have time to make further changes. I have done some code cleaning but I have not re-ran all the experiments in the paper to make sure it works. If you have any questions about the code, you can email Simon Oya (simon.oya@uwaterloo.ca)

The code can be used as a framework to run the experiments, or the attack files can be taken independently to incorporate the attacks in other evaluations (see a description of the input variables of the attacks below).
The overall evaluation framework is the following:
1) Create an ``ExpParams`` object, defined in ``exp_params.py``.
This object defines an experiment, and contains the general parameters, attack parameters, and defense parameters.
2) Call ``run_experiment`` inside ``experiment.py``, passing an ``ExpParams`` object as an argument.
This script runs the experiment with the parameters specified in the object.
It returns the accuracy values, unless called with the debug option active; in that case, it prints the accuracy results.

The file ``debug.py`` contains examples of how to run basic experiments.
The file ``defense.py`` contains the function that generates the adversary observations given the real traces and the defense parameters, while the ``attacks`` folder contains the attacks implementation.

The file ``manager.py`` defines a data structure to queue experiments (calling ``add_to_manager.py``), to run them (calling ``run_from_manager.py``), and to print the results.
The plot scripts call a manager database to load results and print them.
This might be clunky, but using this manager is optional: feel free to program your own experiment manager.

Below there is a description of the ExpParams class, attacks, and datasets. (**TODO:** in progress! I want to add defense description).


# ExpParams

The file ``exp_params.py`` defines the ``ExpParams`` class.
This class has three attributes: ``gen_params`` (general parameters), ``att_params`` (attack parameters), and ``def_params`` (defense parameters).
These attributes are dictionaries.
Their keys and possible values are specified below.

### General parameters (keys and values)
- ``'dataset'``: specifies the dataset name to load. Now we have ``'enron-full'``, ``'lucene'``, ``'movie-plots'``, ``'articles1'``, and ``'bow-nytimes'`` for the regular experiments. We have ``'wiki_pri'``, ``'wiki_pol'``, ``'wiki_cry'``, ``'wiki_act'``, and ``'wiki_sec'`` for the Wikipedia (Markov) experiments.
- ``'nkw'``: Number of keywords to take for this experiment (integer).
- ``'nqr``: Total number of queries that the client sends (integer).
- ``'ndoc'``: Total number of documents for the client's AND auxiliary datasets (integer). It can also be the string ``'full'``, denoting that all documents are used. (**TODO:** change this so that it's just the client's dataset)
- ``'freq'``: Source for frequency information
  - ``'file'``: loaded from a file with the dataset name
  - ``'zipf'``: frequencies follow a Zipfian distribution
  - ``'zipfs<int>'``: follows a Zipfian distribution with ``<int>`` as a shift.
  - ``'none'``: no frequency info (the real frequencies follow a uniform distribution).
- ``'mode_ds'``: mode for splitting the dataset into the client's and auxiliary's datasets. (**TODO:** I want to change this)
  - ``'same'``: they are the same.
  - ``'common<int>'``: they adversary's dataset has ``<int>`` percent of documents in common with the client's dataset.
  - ``'split<int>'``: the adversary's dataset is ``<int>`` percent of the dataset, and the remaining is the client's dataset
  - ``'splitn<int>'``: the adversary's dataset has ``<int>`` documents, and the remaining are the client's dataset.
- ``'mode_kw'``: determines how the ``nkw`` keywords are chosen from all the dataset's keywords.
  - ``'top'``: the top most popular keywords are chosen.
  - ``'rand'``: the keywords are chosen at random.
- ``'mode_fs'``: mode for splitting the frequencies into the client's (real frequencies) and the adversary's frequencies.
  - ``'same'``: the client and adversary use the same frequencies.
  - ``'past'``: the adversary gets outdated frequencies.
- ``'mode_query'``: determines how the keywords of each query are chosen.
  - ``'idd'``: queries are independent and identically distributed.
  - ``'markov'``: queries follow a Markov model.
  - ``'each'``: queries are for distinct keywords.
- ``'known_queries'``: number of distinct keyword queries that the adversary gets as ground-truth information (integer). 
  
### Attack parameters (keys and values)

This is a list, for each attack, of its parameters and default values of each, in dictionary format (**TODO**: add detailed description of this):
- ``'freq'``: ``{}``,
- ``'ikk'``: ``{'naive': False, 'unique': True, 'cooling': 0.9999}``,
- ``'graphm'``: ``{'naive': False, 'alpha': 0.5}``,
- ``'sap'``: ``{'naive': False, 'alpha': 0.5}``,
- ``'umemaya'``: ``{'naive': False}``,
- ``'ihop'``: ``{'naive': False, 'mode': 'Vol', 'pfree': 0.25, 'niters': 1000}``,
- ``'fastpfp'``: ``{'naive': False, 'alpha': 0.5}``,

IHOP can also get ``niter_list``; this is a "hack" to return the accuracy of the attack at different values of ``'niters'`` instead of running the attack separately for each value of ``'niters'``.

### Defense parameters (keys and values)

Likewise, this is a list, for each defense, of its parameters and default values:

- ``'none'``: ``{}``,
- ``'clrz'``: ``{'tpr': 1.0, 'fpr': 0.0}``,
- ``'osse'``: ``{'tpr': 1.0, 'fpr': 0.0}``,
- ``'pancake'``: ``{}``,



# Attacks

All the attacks receive three dictionary inputs: ``obs``, ``aux``, and ``exp_params``.
* ``obs``: dictionary with the observations. It contains the following keys:
  * ``'ndocs``: number of documents in the dataset
  * ``'trace_type'``: string specifying which of the trace types the defense leaks: it can be either ``'ap_unique'``, ``'ap_osse'``, or ``'tok_vol'``. 
  * ``'traces'``: is a list with the leakage from the queries. The format of the traces depends on the defense. So far we have three different formats:
      * ``'ap_unique'``: each query is a tuple ``(token_id, documents_retrieved)``, where ``token_id`` is an integer that represents the search pattern leakage, and the ``documents_retrieved`` are a list with the ids of the documents returned for that query. ``'none'`` and ``'clrz'`` defenses use this leakage.
      * ``'ap_osse'``: each query is a list with the ids of the documents returned for that query (search pattern is not leaked). This is the leakage format for ``'osse'`` defense.
      * ``'tok_vol'``: each query is a tuple ``(token_id, volume)``, where ``token_id`` is an integer that represents the search pattern leakage and ``volume`` is the query volume. 
      This is the model where there is no co-occurrence leakage, so only raw volume matters. ``'pancake'``, ``'seal'`` and ``'ppyy'`` use this leakage. (**TODO:** add SEAL and PPYY!)
* ``aux``: dictionary with auxiliary information. Some of these keys are optional.
  * ``'dataset'``: dataset with *non-indexed* documents for statistical-based attacks (list of documents, where each document is a list of keyword ids)
  * ``'keywords'``: chosen keyword indices for this run (this determines the alphabet that the adversary sees, only that it's in terms of integers or keyword IDs, not in terms of strings)
  * ``'mode_query'``: this is ``'iid'`` when the user sends iid queries, ``'markov'`` when it sends markov-based queries, and ``'each'`` when it queries each keyword once.
  * ``'frequencies'``: frequency information for statistical-based attacks. If ``'mode_query'=='iid'``, this is an np.array of size ``len(keywords)`` with the frequencies of each keyword. If ``'mode_query'=='markov'``, this is a Markov matrix, and if ``'mode_query'=='each'`` this is ``None``.
    (**TODO:** Ensure this is the case, right now it's a bit clunky as it depends on other general parameters.)
  * ``'ground_truth_queries'``: correspondence between some queries and their underlying keyword (list with (query_position, kw_id) tuples).
* ``exp_params``: additional experiment parameters. In the code, we send the whole experiment configuration, cause some attacks might need this information (**TODO:** make this more specific, we might not need everything, some info is already in ``aux``, etc.)

**TODO:** I want to separate between non-indexed dataset and ground-truth dataset in the auxiliary information. Now the ``'dataset'`` field will have non-indexed documents when the ``'mode_ds'`` is ``'split'`` but ground-truth documents when the split mode is ``'same'`` or ``'common'``.

# Datasets and processing

We have three folders: 
1) ``datasets_raw``: contains the datasets (and possibly auxiliary files), as downloaded from the web.
2) ``datasets_pre``: contains pre-processed datasets that meet a common format.
Each dataset in this folder is a pickle file with a tuple ``(dataset, vocabulary)``:
   1) ``dataset``: a list of documents, each document is a list of kwID's
   2) ``vocabulary``: a list of keywords (in kwID order)
3) ``datasets_pro``: contains processed datasets, that are pickle files with a tuple ``(dataset, stems, aux)``:
   1) ``dataset``: a list of documents, each document is a list of kwID's (filtered)
   2) ``stems``: stems in kwID order. These are stemmed words, excluding stopwords and words of length smaller than two or greater than twenty.
   3) ``aux``: a dictionary with auxiliary info. In the new dataset, the keyword ``'stems_to_words'`` gives a dictionary from stems to the list of keywords that resulted in that stem. We can use this to get the query frequencies.

## Processing

After downloading the datasets as explained in ``datasets_raw/download_raw_datasets.sh``, we pre-process them using the function ``preprocess_raw_dataset(dataset_name)`` in ``process_datasets.py``.
Then, we process the pre-processed datasets using the function ``process_pre_dataset(dataset_name)`` in ``process_datasets.py``.
The final dataset is a list (of documents) of lists (keywords), where the keywords belong to the keyword universe.

The respository includes an example of a pre-processed dataset (``datasets_pre/enron-full.pkl``) and all the processed datasets (under ``datasets_pro/``) except for 'NYTimes', since its size is larger than 100MB.

### Downloading query frequencies:
The processed datasets in the repository already contain the query frequencies. If loading the processed dataset like this,
```
with open(path_to_pro_dataset, "rb") as f:
        dataset, stems, aux = pickle.load(f)
```
the variable ``aux['trends']`` contains a 3000 x 52 matrix where the i-th row contains the query frequency of the i-th keyword (stem) for each of the 52 weeks of 2020.
        
To generate new/more query frequencies from Google Trends, please check functions ``get_frequencies_from_google_trends`` and ``add_frequency_trends_information_to_dataset`` in ``process_datasets.py``.


## Datasets

### Enron
The original Enron dataset is ``enron_mail_20150507.tar.gz``, downloaded from https://www.cs.cmu.edu/~enron/. When processed, we call it ``enron-full``.

### Lucene
User posts downloaded from http://mail-archives.apache.org/mod_mbox/lucene-java-user. We call the dataset with all data from years 2002 to 2020 ``lucene``.

### Bag of words
Downloaded from https://archive.ics.uci.edu/ml/datasets/bag+of+words. I think we will use the NYTimes dataset only (``docword.nytimes.txt.gz`` has the dataset with keyword ID's, ``vocab.nytimes.txt`` has the alphabet, when we process it we call it ``bow-nytimes.pkl``)

### News dataset
Downloaded from https://www.kaggle.com/snapcrack/all-the-news/version/4. I am trying ``articles1.csv``.

### Movie plots dataset
Downloaded from http://www.cs.cmu.edu/~ark/personas/

### Book summaries dataset
Downloaded from http://www.cs.cmu.edu/%7Edbamman/booksummaries.html
(not used in the paper since it didn't add anything new)
