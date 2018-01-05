#!/bin/bash
set -eo pipefail

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading data..."
mkdir $BASEDIR/data
wget -q http://norvig.com/big.txt -O $BASEDIR/data/big.txt # Peter Norvig's big.txt for preprocessing scripts
wget -q http://gautam.cc/docs/quora-questions/train.csv.zip -O $BASEDIR/data/train.csv.zip
wget -q http://gautam.cc/docs/quora-questions/sample_submission.csv.zip -O $BASEDIR/data/sample_submission.csv.zip
wget -q http://gautam.cc/docs/quora-questions/test.csv.zip -O $BASEDIR/data/test.csv.zip
unzip $BASEDIR/data/train.csv.zip; rm -rf $BASEDIR/data/train.csv.zip
unzip $BASEDIR/data/test.csv.zip; rm -rf $BASEDIR/data/test.csv.zip
unzip $BASEDIR/data/sample_submission.csv.zip; rm -rf $BASEDIR/data/sample_submission.csv.zip
mv $BASEDIR/sample_submission.csv $BASEDIR/test.csv $BASEDIR/train.csv $BASEDIR/data/

echo "Downloading GloVe Vectors..."
mkdir $BASEDIR/glove
wget -c http://nlp.stanford.edu/data/glove.840B.300d.zip -O $BASEDIR/glove/glove.840B.300d.zip
unzip $BASEDIR/glove/glove.840B.300d.zip; rm -rf $BASEDIR/glove/glove.840B.300d.zip
mv $BASEDIR/glove.840B.300d.txt $BASEDIR/glove/

echo "Installing dependencies..."
pip install -r $BASEDIR/requirements.txt
