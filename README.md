# Quora Pairs
[Can you identify question pairs that have the same intent?](https://www.kaggle.com/c/quora-question-pairs). Heavily inspired by [Quora's Kaggle Challenge](https://www.kaggle.com/c/quora-question-pairs), my solution uses a simple feed-forward neural network written in TensorFlow to classify two questions as having the same intent or not. This is very important for systems that utilize user-generated questions (i.e. Quora, Edmodo's AskMo, etc). 

The neural network can be easily repurposed for other natural language classification tasks, though the deduplication component could be improved using an LSTM network. The current architecture achieves approximately ~81% precision on Quora's data.

## Installation
Build the component and install all of the necessary vectors, datasets, and dependencies.
```
./build.sh
```

## Usage
### Training
To train a new model simply run:
```
python train.py
```
Although not required, having an NVIDIA GPU to train with often speeds up iteration and the development cycle. The pre-trained models were trained using an NVIDIA Tesla K80.

### Evaluation
To evaluate and test a saved model (build this component and you can try out pre-trained models stored in ```save/```), run the following script:
```
python evaluate.py "What is the meaning of life?" "What does life really mean?"
```
The evaluation script will take a few seconds to initialize the TensorFlow model and load the saved ```model.ckpt``` into memory. The evaluation script can also be imported as a Python module for faster in-memory evaluation.
