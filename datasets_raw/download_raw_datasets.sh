#!/bin/sh

# Download enron (original)
#curl -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz

# Download bag of words
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.kos.txt.gz
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.pubmed.txt.gz
#
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.enron.txt
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.kos.txt
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nips.txt
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nytimes.txt
#curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.pubmed.txt

## Apache Lucene Java-user
#mkdir lucene-java-user
#cd lucene-java-user
#for year in {2002..2020}; do
#    for month in {01..12}; do
#        wget "http://mail-archives.apache.org/mod_mbox/lucene-java-user/${year}${month}.mbox"
#    done
#done

curl -O http://www.cs.cmu.edu/%7Edbamman/data/booksummaries.tar.gz
tar xzvf booksummaries.tar.gz
rm booksummaries.tar.gz


curl -O http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
tar xzvf MovieSummaries.tar.gz
rm MovieSummaries.tar.gz