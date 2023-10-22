import os
import numpy as np
import zipfile
import requests

def gloveEmb_import(url_path="https://nlp.stanford.edu/data"):
    url = f"{url_path}/glove.6B.zip"
    cache_dir = "glove_embedding_cache"
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_path = os.path.join(cache_dir, "glove.6B.zip")
    
    if not os.path.exists(cache_path):
        response = requests.get(url)
        with open(cache_path, "wb") as f:
            f.write(response.content)
    
    with zipfile.ZipFile(cache_path, "r") as zip_ref:
        zip_ref.extract("glove.6B.100d.txt", cache_dir)
    
    embeddings_index = {}
    with open(os.path.join(cache_dir, "glove.6B.100d.txt"), encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    
    return embeddings_index







import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def encode_ViralSeq(trainingSet, embeddings_index):
    v_text = trainingSet["Virus_Seq"].str.cat(sep=" ").split()
    
    max_words = 20  # Considering only the standard amino acids
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(v_text)
    
    v_sequences = tokenizer.texts_to_sequences(v_text)
    maxlen = 1000  # We will cut sequences after 1000 words
    data_v = pad_sequences(v_sequences, maxlen=maxlen)
    
    word_index = tokenizer.word_index
    embedding_dim = 100
    embedding_matrix_v = np.zeros((max_words, embedding_dim))
    
    for word, index in word_index.items():
        if index < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_v[index] = embedding_vector
    
    return {"embedding_matrix_v": embedding_matrix_v, "data_v": data_v}



import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def encode_HostSeq(trainingSet, embeddings_index):
    h_text = trainingSet["Human_Seq"].str.cat(sep=" ").split()
    
    max_words = 20  # Considering only the standard amino acids
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(h_text)
    
    h_sequences = tokenizer.texts_to_sequences(h_text)
    maxlen = 1000  # We will cut sequences after 1000 words
    data_h = pad_sequences(h_sequences, maxlen=maxlen)
    
    word_index = tokenizer.word_index
    embedding_dim = 100
    embedding_matrix_h = np.zeros((max_words, embedding_dim))
    
    for word, index in word_index.items():
        if index < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_h[index] = embedding_vector
    
    return {"embedding_matrix_h": embedding_matrix_h, "data_h": data_h}



import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, concatenate, Dense
from keras.optimizers import Adam

def load_pretrained_model(input_dim=20, output_dim=100, filters_layer1CNN=32, kernel_size_layer1CNN=16,
                          filters_layer2CNN=64, kernel_size_layer2CNN=7, pool_size=30, layer_lstm=64, units=8,
                          metrics=["AUC"], filepath="path_to_pretrained_model.h5"):
    # Load embeddings from CSV files
    embedding_matrix_v = pd.read_csv("path_to_viral_embedding.csv").values
    embedding_matrix_h = pd.read_csv("path_to_host_embedding.csv").values

    # Define the input layers
    seq1 = Input(shape=(1000,), name="viral_seq")
    seq2 = Input(shape=(1000,), name="host_seq")

    # Define the embedding layers for viral and host sides
    embedding_layer_v = Embedding(input_dim=input_dim, output_dim=output_dim,
                                  weights=[embedding_matrix_v], input_length=1000, trainable=False)
    embedding_layer_h = Embedding(input_dim=input_dim, output_dim=output_dim,
                                  weights=[embedding_matrix_h], input_length=1000, trainable=False)

    # Apply the embedding layers to input sequences
    viral_embedded = embedding_layer_v(seq1)
    host_embedded = embedding_layer_h(seq2)

    # Define the CNN and LSTM layers
    filters_layer1CNN = 32
    kernel_size_layer1CNN = 16
    filters_layer2CNN = 64
    kernel_size_layer2CNN = 7
    pool_size = 30
    layer_lstm = 64

    conv1_viral = Conv1D(filters=filters_layer1CNN, kernel_size=kernel_size_layer1CNN,
                        activation='relu', strides=2)(viral_embedded)
    conv2_viral = Conv1D(filters=filters_layer2CNN, kernel_size=kernel_size_layer2CNN,
                        activation='relu', strides=2)(conv1_viral)
    max_pool_viral = MaxPooling1D(pool_size=pool_size)(conv2_viral)
    bidirectional_lstm_viral = Bidirectional(LSTM(layer_lstm))(max_pool_viral)
    dropout_viral = Dropout(rate=0.3)(bidirectional_lstm_viral)

    conv1_host = Conv1D(filters=filters_layer1CNN, kernel_size=kernel_size_layer1CNN,
                        activation='relu', strides=2)(host_embedded)
    conv2_host = Conv1D(filters=filters_layer2CNN, kernel_size=kernel_size_layer2CNN,
                        activation='relu', strides=2)(conv1_host)
    max_pool_host = MaxPooling1D(pool_size=pool_size)(conv2_host)
    bidirectional_lstm_host = Bidirectional(LSTM(layer_lstm))(max_pool_host)
    dropout_host = Dropout(rate=0.3)(bidirectional_lstm_host)

    # Concatenate both sides
    merged = concatenate([dropout_viral, dropout_host], axis=-1)

    # Define MLP layers
    dense1 = Dense(units=units, activation='relu')(merged)
    output = Dense(units=1, activation='sigmoid')(dense1)

    # Create the model
    model = Model(inputs=[seq1, seq2], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    # Load pre-trained weights
    model.load_weights(filepath)

    return model



import numpy as np
import pandas as pd
from keras.models import load_model

def pred_interactions(url_path="https://nlp.stanford.edu/data/glove.6B.zip", testing_set, model_path):
    # Load GloVe embeddings
    embeddings_index = gloveEmb_import(url_path)

    # Encode viral and host sequences
    viral_embedding = encode_viral_seq(testing_set, embeddings_index)
    host_embedding = encode_host_seq(testing_set, embeddings_index)

    # Combine the embeddings
    x_pred = np.hstack((viral_embedding['data_v'], host_embedding['data_h']))

    # Load the pre-trained model
    model = load_model(model_path)

    # Predict interactions
    prob = model.predict([x_pred[:, :1000], x_pred[:, 1000:2000])

    return prob


