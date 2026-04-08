# Natural Language Processing with Convolutional Neural Networks (CNNs)

This project performs **sentiment analysis on movie reviews** using various Convolutional Neural Network (CNN) architectures. The dataset used is the **Movie Review Dataset (rt-polaritydata)**, which consists of positive and negative reviews. Multiple CNN architectures are experimented with to determine the most effective model for this task.

## Project Highlights
- Implemented multiple CNN architectures for sentiment classification.
- Used pre-trained Word2Vec embeddings from the Google News dataset.
- Tuned hyperparameters such as learning rate, weight decay, and number of epochs.
- Evaluated models on a test set and reported **training loss** and **final test accuracy**.

## File Structure
- `NLP_Analysis.py`  
  - Comprehensive script for performing sentiment analysis with CNNs.  
  - Steps included:
    - Downloading and extracting the dataset.
    - Loading and preprocessing data.
    - Tokenizing text and creating vocabulary based on word frequency.
    - Converting reviews into sequences of token IDs.
    - Splitting data into training and testing sets.
    - Creating an embedding matrix using pre-trained Word2Vec embeddings.
    - Defining and experimenting with **four different CNN architectures** (varying convolutional layers, fully connected layers, activation functions, and optimizers).
    - Training models and printing training loss for each epoch and final test accuracy.
  
- `rt-polarity.neg`: Contains negative movie reviews.
- `rt-polarity.pos`: Contains positive movie reviews.

## Dataset
The dataset used is the **rt-polaritydata** from Cornell University, consisting of positive and negative movie reviews. It can be downloaded using the following commands:

```bash
wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
tar -xvzf rt-polaritydata.tar.gz
```

## Usage
To run the code, use the following command
```
python NLP_Analysis.py
```

## Requirements:

- Python 3.x
- PyTorch
- TorchText
- spaCy
- Gensim
- scikit-learn
- requests
- portalocker
- numpy
- gzip
- shutil



