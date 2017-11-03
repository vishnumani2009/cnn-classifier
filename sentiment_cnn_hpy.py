"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

Readaptation based on code by https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py

"""
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC,SVC
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
import sys,keras
import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from sklearn.metrics import precision_recall_fscore_support
np.random.seed(0)
from sklearn.model_selection import ParameterGrid
# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static


def load_data(run):
    x, y, vocabulary, vocabulary_inv_list, train_len, dev_len, test_len = data_helpers.load_data(run)
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_dev = x[train_len:train_len + dev_len]
    y_dev = y[train_len:train_len + dev_len]
    x_test = x[train_len + dev_len:]
    y_test = y[train_len + dev_len:]

    print(x_train.shape, x_dev.shape, x_test.shape)
    return x_train, y_train, x_dev, y_dev, x_test, y_test, vocabulary_inv

def testmodel(pmgrid):
# Model Hyperparameters
    #{'num_filters': 75, 'run': 5, 'embedding=dim': 150}
    embedding_dim = pmgrid["embedding_dim"]
    filter_sizes = (3, 5, 8)
    num_filters = pmgrid["num_filters"]
    dropout_prob = (0.5, 0.8)
    hidden_dims = 50

    # Training parameters
    batch_size = 64
    num_epochs = 10

    # Prepossessing parameters
    sequence_length = 400
    max_words = 5000

    # Word2Vec parameters (see train_word2vec)
    min_word_count = 1
    context = 10
    run=pmgrid["run"]

    #
    # ---------------------- Parameters end -----------------------



    # Data Preparation
    #for run in range(1,6):
    print("Load data..."+str(run))
    x_train, y_train,x_dev,y_dev, x_test, y_test, vocabulary_inv = load_data(run)
    # for i in range(len(x_train[0])):
    #     print(vocabulary_inv[i])#get words of voca
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    y_dev = keras.utils.to_categorical(y_dev, 2)


    if sequence_length != x_test.shape[1]:
        # print("Adjusting sequence length for actual size")
        sequence_length = x_test.shape[1]

    # print("x_train shape:", x_train.shape)
    # print("x_test shape:", x_test.shape)
    # print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Prepare embedding layer weights and convert inputs for static model
    # print("Model type is", model_type)
    if model_type in ["CNN-non-static", "CNN-static"]:
        embedding_weights = train_word2vec(np.vstack((x_train, x_dev,x_test)), vocabulary_inv, num_features=embedding_dim,
                                           min_word_count=min_word_count, context=context)
        if model_type == "CNN-static":
            x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
            x_dev = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_dev])
            x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
            # print("x_train static shape:", x_train.shape)
            # print("x_test static shape:", x_test.shape)

    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # print("x_train static shape:", x_train.shape)
    # print("x_test static shape:", x_test.shape)
    # Build model
    if model_type == "CNN-static":
        input_shape = (sequence_length, embedding_dim)
    else:
        input_shape = (sequence_length,)

    model_input = Input(shape=input_shape)

    # Static model does not have embedding layer
    if model_type == "CNN-static":
        z = model_input
    else:
        z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

    z = Dropout(dropout_prob[0])(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(2, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Initialize weights with word2vec
    if model_type == "CNN-non-static":
        weights = np.array([v for v in embedding_weights.values()])
        # print("Initializing embedding layer with word2vec weights, shape", weights.shape)
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])
    # print(model.summary())
    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,validation_data=(x_dev, y_dev), verbose=0)
    ypred=(model.predict(x_dev).argmax(axis=-1))

    score=(precision_recall_fscore_support(y_dev.argmax(axis=-1),ypred))
    print(score)
    return score,model,pmgrid,x_test,y_test

def main():
    # embedding_dim = 50
    # filter_sizes = (3, 5, 8)
    # num_filters = 50
    # dropout_prob = (0.5, 0.8)
    # hidden_dims = 50
    #
    # # Training parameters
    # batch_size = 64
    # num_epochs = 50
    #
    # # Prepossessing parameters
    # sequence_length = 400
    # max_words = 5000
    #
    # # Word2Vec parameters (see train_word2vec)
    # min_word_count = 1
    # context = 10
    # run=1
    #
    #
    # param_grid={"embedding=dim":[50,75,100,150],"filter_sizes":[(3,5,8),(3,3,3),(5,5,5),(5,5,8)],
    #             "num_filters":[25,50,100,75],"dropout_prob":[(0.5,0.5),(0.5,0.8)],"hidden_dims":[25,50,75,100],
    #             "batch_size":[16,32,64,128],"epochs":[25,50,100],"sequence_length":[400,300,500,600],
    #             "max_words":[5000,4000,6000],"context":[10,11,15,20],"run":[1,2,3,4,5]}
    # param_grid = {"embedding=dim": [50, 75, 100, 150],
    #               "num_filters": [25, 50, 100, 75],
    #             "epochs": [25, 50, 100], "sequence_length": [400, 300, 500, 600],
    #                "context": [10, 11, 15, 20], "run": [1, 2, 3, 4, 5]}
    param_grid = {"embedding_dim": [50, 75, 100, 150],
                  "num_filters": [25, 50, 100, 75],
                  "run": [1]}
    bscore=0
    bmodel=None
    bparam=None
    for pm in list(ParameterGrid(param_grid)):
        print(pm)
        score,model,param,xt,yt=testmodel(pm)
        if score[2][1]>bscore:
            bscore=score[2][1]
            bmodel=model
            bparam=param
            print(bscore,bparam)
        ypred=(model.predict(xt).argmax(axis=-1))
        print(classification_report(yt.argmax(axis=-1),ypred))

main()