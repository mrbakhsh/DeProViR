# Import necessary libraries
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, concatenate
from tensorflow.keras.models import Model
import pandas as pd

# Load embedding matrix for the viral side (actual path: DeProViR/extdata/Pre_trainedModel)
embedding_matrix_v = pd.read_csv("path/to/viral_embedding.csv").values

# Define input shape for viral sequence
seq1 = Input(shape=(1000,), name='viral_seq')

# Define model for viral side
m1 = Embedding(input_dim=input_dim, output_dim=output_dim, weights=[embedding_matrix_v], trainable=False, input_length=1000)(seq1)
m1 = Conv1D(filters=filters_layer1CNN, kernel_size=kernel_size_layer1CNN, strides=2, activation='relu')(m1)
m1 = Conv1D(filters=filters_layer2CNN, kernel_size=kernel_size_layer2CNN, strides=2, activation='relu')(m1)
m1 = MaxPooling1D(pool_size=pool_size)(m1)
m1 = Bidirectional(LSTM(units=layer_lstm))(m1)
m1 = Dropout(rate=0.3)(m1)

# Load embedding matrix for the host side (actual path: DeProViR/extdata/Pre_trainedModel)
embedding_matrix_h = pd.read_csv("path/to/host_embedding.csv").values

# Define input shape for host sequence
seq2 = Input(shape=(1000,), name='host_seq')

# Define model for host side
m2 = Embedding(input_dim=input_dim, output_dim=output_dim, weights=[embedding_matrix_h], trainable=False, input_length=1000)(seq2)
m2 = Conv1D(filters=filters_layer1CNN, kernel_size=kernel_size_layer1CNN, strides=2, activation='relu')(m2)
m2 = Conv1D(filters=filters_layer2CNN, kernel_size=kernel_size_layer2CNN, strides=2, activation='relu')(m2)
m2 = MaxPooling1D(pool_size=pool_size)(m2)
m2 = Bidirectional(LSTM(units=layer_lstm))(m2)
m2 = Dropout(rate=0.3)(m2)

# Concatenate the models from viral and host sides
merge_vector = concatenate([m1, m2])

# Define output layer
out = Dense(units=units, activation='relu')(merge_vector)
out = Dense(units=1, activation='sigmoid')(out)

# Define and compile the merged model
trainedModel = Model(inputs=[seq1, seq2], outputs=out)
trainedModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics])

# Load pre-trained model weights (actual path: DeProViR/extdata/Pre_trainedModel)
filepath = "path/to/pre_trained_glove_model_PubTrained_final_cv.h5"
trainedModel.load_weights(filepath)


