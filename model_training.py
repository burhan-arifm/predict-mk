from datetime import datetime, timedelta
from math import sqrt
from os import system
from pathlib import Path
from time import time

from keras import Model
from keras.layers import (Add, BatchNormalization, Dense, Dropout, Embedding,
                          Flatten, Input, LayerNormalization, LeakyReLU,
                          MultiHeadAttention, Multiply, Reshape, StringLookup,
                          concatenate)
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adagrad
from keras.utils.vis_utils import plot_model
from matplotlib.pyplot import legend, plot, savefig, title, xlabel, ylabel
from numpy.random import rand
from pandas import DataFrame, read_csv
from tensorflow import (data, expand_dims, float32, keras, range, string,
                        strings, unstack)

# Data Condition
# 1. mahasiswa.csv: column ['nim (11X4050XXX)','nama','jenis_kelamin (L/P)']
# 3. matkul.csv: reformatted (JURXXXX/JURXXXXXX to JUR17XXX), column ['kode_mk (JUR17XXX)','mata_kuliah','semester (1-7)']
# 4. nilai-20XX.csv: column ['nim (11X4050XXX)','kode_mk (JUR17XXX)','nilai (A-F,T)']

# Load CSV data to Pandas DataFrame
# in this step, we're gonna load all the data to pandas dataframe
nilai = read_csv(
    'datasets/nilai_firsttake.csv',
    delimiter=';',
    names=['nim', 'kode_mk', 'nilai']
)
matkul = read_csv(
    'datasets/matkul_headless.csv',
    delimiter=';',
    names=['kode_mk', 'mata_kuliah', 'semester',
           'sifat', 'lokasi', 'bobot_sks']
)
mahasiswa = read_csv(
    'datasets/mahasiswa_headless.csv',
    delimiter=';',
    names=['nim', 'nama_mahasiswa', 'jenis_kelamin']
)

mahasiswa['nim'] = mahasiswa['nim'].apply(lambda x: f'mhs_{x}')
nilai['nim'] = nilai['nim'].apply(lambda x: f'mhs_{x}')
nilai['nilai'] = nilai['nilai'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'T'], [
    4, 3, 2, 1, 0, 0, 0])
nilai['mk_nilai'] = nilai['kode_mk'].astype(
    str) + '_' + nilai['nilai'].astype(str)
nilai_smt_7 = nilai[nilai['kode_mk'].str.contains('JUR177')]
nilai_smt_1_6 = nilai.loc[~nilai['kode_mk'].str.contains('JUR177')]


def group_dataset(dataset):
    dataset_group = dataset.groupby('nim')
    dataset_data = DataFrame(
        data={
            'nim': list(dataset_group.groups.keys()),
            'mk_nilai': list(dataset_group.mk_nilai.apply(list))
        }
    )

    return dataset_data


nilai_smt_7 = group_dataset(nilai_smt_7)
nilai_smt_1_6 = group_dataset(nilai_smt_1_6)

nilai_data = nilai_smt_1_6.merge(
    nilai_smt_7, on='nim', suffixes=('_1_6', '_7'))

del mahasiswa['nama_mahasiswa']

sequence_length = 5
step_size = 3

fmt = '%Y%m%d-%H.%M.%S%z'
path = fr'experiments/{datetime.now().strftime(fmt)}_seq{sequence_length}_step{step_size}'
Path(path).mkdir(parents=True, exist_ok=True)


def create_sequences(values, window_size, step_size):
    sequences = []
    mk_nilai_1_6, mk_nilai_7 = values
    for mk_nilai in mk_nilai_7:
        start_index = 0
        end_index = 0
        while end_index < len(mk_nilai_1_6):
            end_index = start_index + window_size - 1
            seq = mk_nilai_1_6[start_index:end_index]
            if len(seq) < window_size - 1:
                seq = mk_nilai_1_6[-(window_size - 1):]
            seq.append(mk_nilai)
            sequences.append(seq)
            start_index += step_size

    return sequences


nilai_data['mk_nilai'] = nilai_data.apply(lambda x: create_sequences(
    (x['mk_nilai_1_6'], x['mk_nilai_7']), sequence_length, step_size), axis=1)

del nilai_data['mk_nilai_1_6'], nilai_data['mk_nilai_7']

random_selection = rand(len(nilai_data.index)) <= 0.8
train_data = nilai_data[random_selection]
test_data = nilai_data[~random_selection]
predict_data = test_data.iloc[[0]]


def finalize_dataframe(dataframe):
    dataframe_transformed = dataframe[['nim', 'mk_nilai']].explode(
        'mk_nilai', ignore_index=True)
    # dataframe_transformed = dataframe_transformed.join(
    #     mahasiswa.set_index('nim'), on='nim')

    def split_column(values):
        value_1 = []
        value_2 = []
        for string in values:
            value = string.split('_')
            value_1.append(value[0])
            value_2.append(value[1])

        return value_1, value_2

    dataframe_transformed['mk_nilai'] = dataframe_transformed['mk_nilai'].apply(
        lambda x: split_column(x))
    dataframe_transformed[['seq_matkul', 'seq_nilai']] = DataFrame(
        dataframe_transformed['mk_nilai'].to_list(), index=dataframe_transformed.index)

    dataframe_transformed.seq_matkul = dataframe_transformed.seq_matkul.apply(
        lambda x: ','.join(x))
    dataframe_transformed.seq_nilai = dataframe_transformed.seq_nilai.apply(
        lambda x: ','.join(x))
    del dataframe_transformed['mk_nilai']

    return dataframe_transformed


train_data = finalize_dataframe(train_data)
predict_data = finalize_dataframe(predict_data)
test_data = finalize_dataframe(test_data)

train_data.to_csv(f'{path}/train_data.csv',
                  index=False, sep='|', header=False)
test_data.to_csv(f'{path}/test_data.csv', index=False, sep='|', header=False)
predict_data.to_csv(f'{path}/predict.csv', index=False, sep='|', header=False)

CSV_HEADER = list(train_data.columns)
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    'nim': list(mahasiswa.nim.unique()),
    'kode_mk': list(matkul.kode_mk.unique()),
    'jenis_kelamin': list(mahasiswa.jenis_kelamin.unique())
}
MHS_FEATURES = ['jenis_kelamin']
MATKUL_FEATURES = ['bobot_sks']


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    def process(features):
        kode_mk_string = features['seq_matkul']
        seq_matkul = strings.split(kode_mk_string, ',').to_tensor()

        features['target_matkul'] = seq_matkul[:, -1]
        features['seq_matkul'] = seq_matkul[:, :-1]

        nilai_string = features['seq_nilai']
        seq_nilai = strings.to_number(
            strings.split(nilai_string, ',')).to_tensor()

        target = seq_nilai[:, -1]
        features['seq_nilai'] = seq_nilai[:, :-1]

        return features, target

    dataset = data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim='|',
        shuffle=shuffle
    ).map(process)

    return dataset


def create_model_inputs():
    return {
        'nim': Input(name='nim', shape=(1, ), dtype=string),
        'seq_matkul': Input(name='seq_matkul', shape=(sequence_length - 1, ), dtype=string),
        'target_matkul': Input(name='target_matkul', shape=(1, ), dtype=string),
        'seq_nilai': Input(name='seq_nilai', shape=(sequence_length - 1, ), dtype=float32),
        # 'jenis_kelamin': Input(name='jenis_kelamin', shape=(1, ), dtype=string),
    }


def encode_input_features(inputs, include_nim=True, include_mhs_features=True, include_matkul_features=True):
    encoded_transformer_features = []
    encoded_other_features = []
    other_feature_names = []

    if include_nim:
        other_feature_names.append('nim')
    if include_mhs_features:
        other_feature_names.extend(MHS_FEATURES)

    # Encode mahasiswa features
    for feature_name in other_feature_names:
        # Convert string input into integer indices
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        index = StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name])
        # Compute embedding dims
        embedding_dims = int(sqrt(len(vocabulary)))
        # Create an embedding layer with specified dimensions
        embedding_encoder = Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f'{feature_name}_embedding',
        )
        # Convert index values to embedding representations
        encoded_other_features.append(embedding_encoder(index))

    # Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    # Create a matkul embedding encoder
    matkul_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY['kode_mk']
    matkul_embedding_dims = int(sqrt(len(matkul_vocabulary)))
    # Create a lookup to convert string values to integer indices
    matkul_index_lookup = StringLookup(
        vocabulary=matkul_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name='matkul_index_lookup',
    )
    # Create an embedding layer with the specified dimensions
    matkul_embedding_encoder = Embedding(
        input_dim=len(matkul_vocabulary),
        output_dim=matkul_embedding_dims,
        name='matkul_embedding',
    )
    # Create a vector lookup for bobot sks in matkul
    bobot_sks_vector = matkul['bobot_sks'].to_numpy()
    matkul_sks_lookup = Embedding(
        input_dim=bobot_sks_vector.shape[0],
        output_dim=1,
        embeddings_initializer=keras.initializers.Constant(bobot_sks_vector),
        trainable=False,
        name='bobot_sks_vector',
    )
    # Create a processing layer for sks
    matkul_embedding_processor = Dense(
        units=matkul_embedding_dims,
        activation='relu',
        name='process_matkul_embedding_with_bobot_sks',
    )

    # Define a function to encode a given kode_mk
    def encode_matkul(kode_mk):
        # Convert the string input values into integer indices
        matkul = matkul_index_lookup(kode_mk)
        matkul_embedding = matkul_embedding_encoder(matkul)
        encoded_matkul = matkul_embedding

        if include_matkul_features:
            matkul_sks_vector = matkul_sks_lookup(matkul)
            encoded_matkul = matkul_embedding_processor(
                concatenate([matkul_embedding, matkul_sks_vector]))

        return encoded_matkul

    # Encoding target matkul
    target_matkul = inputs['target_matkul']
    encoded_target_matkul = encode_matkul(target_matkul)
    # Encoding seq matkul
    seq_matkul = inputs['seq_matkul']
    encoded_seq_matkul = encode_matkul(seq_matkul)
    # Create positional embedding
    position_embedding_encoder = Embedding(
        input_dim=sequence_length,
        output_dim=matkul_embedding_dims,
        name='position_embedding',
    )
    positions = range(start=0, limit=sequence_length - 1, delta=1)
    encoded_positions = position_embedding_encoder(positions)
    # Retrieve seq nilai to incorporate them into the encoding of the matkul
    seq_nilai = expand_dims(inputs['seq_nilai'], -1)
    # Add the positional encoding to the matkul encodings and multiply them by nilai
    encoded_seq_matkul_with_position_and_nilai = Multiply()(
        [(encoded_seq_matkul + encoded_positions), seq_nilai])

    # Construct the transformer inputs
    for encoded_matkul in unstack(encoded_seq_matkul_with_position_and_nilai, axis=1):
        encoded_transformer_features.append(expand_dims(encoded_matkul, 1))

    encoded_transformer_features.append(encoded_target_matkul)
    encoded_transformer_features = concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_other_features


include_nim = False
include_mhs_features = False
include_matkul_features = False

hidden_units = [1024, 512, 256]
dropout_rate = 0.2
num_heads = 8
num_layers = 1


def encoder_layer(transformer_features):
    features = transformer_features
    dimension = transformer_features.shape[2]

    for _ in range(num_layers):
        # the following block is the (S') based on the paper
        att_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dimension,
            dropout=dropout_rate
        )(features, features)
        x1 = Add()([features, att_output])
        x1 = LayerNormalization()(x1)

        # the following block is the feed forward (F) based on the paper
        x2 = LeakyReLU()(x1)
        x2 = Dense(units=x2.shape[-1])(x2)
        x2 = Dropout(dropout_rate)(x2)
        features = Add()([x1, x2])
        features = LayerNormalization()(features)

    return features


def mlp_layer(features, hidden_units):
    for num_units in hidden_units:
        features = Dense(num_units)(features)
        features = BatchNormalization()(features)
        features = LeakyReLU()(features)
        features = Dropout(dropout_rate)(features)

    return features


def create_model():
    inputs = create_model_inputs()

    # Embedding layer
    transformer_features, other_features = encode_input_features(
        inputs, include_nim, include_mhs_features, include_matkul_features)

    # Transformer layer(s)
    transformer_features = encoder_layer(transformer_features)
    features = Flatten()(transformer_features)

    # Concatenate with other features
    if other_features is not None:
        features = concatenate(
            [features, Reshape([other_features.shape[-1]])(other_features)])

    # MLP layers and loss function
    features = mlp_layer(features, hidden_units)

    # Output layer
    outputs = Dense(units=1)(features)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()
plot_model(model, to_file=f'{path}/model.png', dpi=300, show_shapes=True, show_layer_activations=True,
           show_dtype=True, expand_nested=True)


model.compile(
    optimizer=Adagrad(learning_rate=0.01),
    loss=MeanSquaredError(),
    metrics=[RootMeanSquaredError(name='rmse')],
)

train_dataset = get_dataset_from_csv(
    f'{path}/train_data.csv', shuffle=True, batch_size=256)

system('cls')
start_time = time()
history = model.fit(train_dataset, epochs=75)
finish_time = time()

with open(f'{path}/prediction-result.txt', 'w') as f:
    print(
        f'training time: {timedelta(seconds=(finish_time - start_time))}\n', file=f)

plot(history.history['loss'])
plot(history.history['rmse'])
title('Training Result')
ylabel('Value')
xlabel('Epochs')
legend(['Loss', 'RMSE'], loc='upper left')
savefig(f'{path}/training_graph.png', dpi=300.0, format='png')


test_dataset = get_dataset_from_csv(
    f'{path}/test_data.csv', batch_size=256)
loss, rmse = model.evaluate(
    test_dataset, verbose=0)

with open(f'{path}/prediction-result.txt', 'a') as f:
    print(f"Test Loss: {round(loss, 4)}", file=f)
    print(f"Test RMSE: {round(rmse, 4)}", file=f)

predict_input = get_dataset_from_csv(f'{path}/predict.csv')
prediction = model.predict(predict_input)

with open(f'{path}/prediction-result.txt', 'a') as f:
    print(prediction, file=f)

    reshaped_prediction = prediction.reshape(5, int(prediction.shape[0]/5))
    print(reshaped_prediction, file=f)


model.save(f'{path}/saved-model')
