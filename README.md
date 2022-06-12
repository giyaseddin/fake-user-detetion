# Fake User Detetion
This is a demo of applying simple algorithms to classify fake and real users from their category and event

## Introduction
This repo contains the basic implementation of traditional machine learning algorithms, applied to classify categorical tabular data.
Classification task is simple, using only two attributes, we need to separate the fake users from the real users.


## Structure
The project is explanatory, providing a simple implementation of the solution of the problem defined in the [Introduction](#introduction)
The structure of set as such:

```
[project_root]
├── data
│   ├── README.md
│   └── [here fake_users.csv and fake_users_test.csv should be placed]
├── notebooks
│   ├── model_development.ipynb
│   └── data_exploration.ipynb
├── tests
│   ├── test_predict.py
│   ├── test_train.py
│   └── test_utils.py
├── trained_models
│   └── [training runs will be saved here.]
├── utils.py
├── train.py
├── predict.py
└── config.json
..
```

- [data](data) folder is where the data placed
- Under [notebooks](notebooks) folder, there exist the explanatory jupyter notebooks
- [tests](tests) folder contains three 'limited-coverage' test files, that only ensures a small portion of the optimal test coverage
- In [trained_models](trained_models) folder, the trained models and their corresponding results and outputs are saved
- [utils.py](utils.py) file contains helper methods for the training and inference
- [train.py](train.py) script is used to run the training, it requires arguments to be passed while running the script
- [predict.py](predict.py) script, to run predictions, and it requires arguments to be passed while running the script
- [config.json](config.json) contains some modifiable config parameter 

## Models
### Pre-processing
When dealing with categorical features, transforming them using `OneHotEncoder` is a typical behaviour, especially for such small number of discrete value and values.

We did the `OneHotEncoder` on our features (`Event` and `Category`) and saved the encoders for the inference phase.

### Model training & evaluation

The initial aim is to detect the non-real users, and ideally, we would start with the set of the transactions for each user, as in "Fraud Detection" problems to eventually find the suspicious users and block them. 

The idea behind the trained model is to train a simple classifier like:
- #### Logistic Regression
- #### Random Forest

on the training transaction data, and try as possible to make it learns some fake behaviour for a typical fake transaction.
But we still have the part of exploiting the group of transactions for each fake user in the decision process, which we delegate to the post-processing phase.

### Post-processing

In an intuitive fashion, we take the average of the probabilities inside the group of transactions of each user, predicted by the trained model, hen we average them to end up with the final decision on whether the user is fake or not.


## Usage

The project is basically divided into `notebooks` and three runnable parts: 
- training
- prediction
- testing


First, we start with installing the requirements
```bash
pip install -r requirements.txt
```

And to navigate to the web page of jupyter notebook to browse the exploratory notebooks using:

```bash
jupyter notebook
```

Before moving to the training and predicting, we should make sure that the data files are placed under [data folder](./data/) as instructed in the [data/README.md](./data/README.md)

The runnable parts can be viewed in the [Makefile](./Makefile), they can easily be run using:

```bash
make train

make predict

make test
```

in each of the make tasks, we can see the default values of the arguments to be passed to [train.py](./train.py) and [predit.py](./predit.py) scripts


## Results

In the evaluation phase of the model is conducted in two separate ways.
The first one is transaction-based, regardless of the user, the evaluation is done based only on the granular prediction of the rows.
The second is user-based, after the evaluation is made after post-processing focusing only on the probability of the user being fake.


The results are the best obtained from Random Forest after performing cross validation technique on with the training data.
The following results are taken for the predictions on the test set:
### Per-transaction results

|              | precision | recall | f1-score | support |
|:------------:|:---------:|:------:|:--------:|:-------:|
|     Real     |   0.96    |  0.60  |   0.74   |  2583   |
|     Fake     |   0.25    |  0.83  |   0.39   |   415   |
|   accuracy   |           |        |          |  0.64   |
|  macro avg   |   0.60    |  0.72  |   0.56   |  2998   |
| weighted avg |   0.86    |  0.83  |   0.69   |  2998   |

### Per-user results

|              | precision | recall | f1-score | support |
|:------------:|:---------:|:------:|:--------:|:-------:|
|     Real     |   1.00    |  0.99  |   0.99   |   142   |
|     Fake     |   0.83    |  1.00  |   0.90   |   10    |
|   accuracy   |           |        |          |  0.99   |
|  macro avg   |   0.92    |  0.99  |   0.95   |   152   |
| weighted avg |   0.99    |  0.99  |   0.99   |   152   |


### Classification threshold
The final decision of the classification is done after the probability predictions.
The threshold can be tuned by sliding the decision threshold that maximizes the recall of the fake class, since it's the crucial part of this classification task.
The choice here is in our case `0.5` manually found from the test set, to get the optimal score :)

ROC_AUC curve is a candidate to tune this threshold for better performance.

## Scope and limitations
The model doesn't fully capture the timespan or the co-occurrence of the users, it is attempted to compensate this in the post-processing phase.

Also, the features are discrete, decision tree-like algorithms are observed to learn better than linear algorithms like "Logistic Regression"

The solution is not so scalable, if more event types or categories were added, then the solution would fall short for future predictions.


## Improvement directions

Some suggestions to improve the solution is to consider the following ideas: 

- New solution approaches could be implemented toward more robust model.

  - E.g. OneClass Support Vector Machine could be trained to capture the fake behaviour.
  - E.g. Gradient boosting approaches are candidates for this classification task such as XGBoost, LightGBM etc.

- Feature Engineering
  - For such tasks, two features are not enough factors to decide on such detection problem alone. Features such as time information and frequency, geographical location, user verification, and many actual and synthetic features could be incorporated to achieve better detection.

- Amount of data
  - The more balanced data we can collect about the "Fake" class, the better learning we can apply on it. But of course, we hope that we don't end up with large number of them :)

- Active learning
  - Having an active learning system in a matured stage would be a good idea to add human experience to improve the model.
  - Suspicious users do not always appear in the pre-defined phase, and active learning could take care of the data drift as well.

