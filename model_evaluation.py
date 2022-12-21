from datetime import datetime, timedelta
from math import sqrt
from operator import itemgetter
from os import system
from pathlib import Path
from time import localtime, strftime, time

from keras import Model
from keras.backend import clear_session
from keras.layers import (Add, BatchNormalization, Dense, Dropout, Embedding,
                          Flatten, Input, LayerNormalization, LeakyReLU,
                          MultiHeadAttention, Multiply, Reshape, StringLookup,
                          concatenate)
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adagrad
from keras.utils.vis_utils import plot_model
from matplotlib.pyplot import (grid, legend, plot, savefig, subplots, title,
                               xlabel, xlim, ylabel, ylim)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from numpy.random import rand
from numpy import average
from pandas import DataFrame, read_csv
from tensorflow import data, expand_dims, float32, keras
from tensorflow import range as tf_range
from tensorflow import string, strings, unstack

# Data Condition
# 1. mahasiswa.csv: column ['nim (11X4050XXX)','nama','jenis_kelamin (L/P)']
# 3. matkul.csv: reformatted (JURXXXX/JURXXXXXX to JUR17XXX), column ['kode_mk (JUR17XXX)','mata_kuliah','semester (1-7)']
# 4. nilai-20XX.csv: column ['nim (11X4050XXX)','kode_mk (JUR17XXX)','nilai (A-F,T)']
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

CSV_HEADER = ['nim', 'seq_matkul', 'seq_nilai']
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    'nim': list(mahasiswa.nim.unique()),
    'kode_mk': list(matkul.kode_mk.unique())
}
MATKUL_FEATURES = ['bobot_sks']

# embedding params
include_nim = False
include_matkul_features = False

# static model params
hidden_units = [1024, 512, 256]
num_heads = 8
dropout_rate = 0.2
learning_rate = 0.01
num_layers = 1
epochs = 30
batch_size = 256
training_dataset_size = 0.8

optimizer_function = Adagrad(learning_rate=learning_rate)
loss_function = MeanSquaredError()
metrics_function = [
    RootMeanSquaredError(name='rmse')
]

date_time = datetime.now()
short_date_fmt = '%Y%m%d'
long_date_fmt = '%A, %d %B %Y'
time_fmt = '%H.%M.%S%z'

main_path = f'eval_pool/{date_time.strftime(short_date_fmt)}/epochs-{epochs}/enc_layers-{num_layers}/{date_time.strftime(time_fmt)}'


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


def finalize_dataframe(dataframe):
    dataframe_transformed = dataframe[['nim', 'mk_nilai']].explode(
        'mk_nilai', ignore_index=True)

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
        'seq_nilai': Input(name='seq_nilai', shape=(sequence_length - 1, ), dtype=float32)
    }


def encode_input_features(inputs, include_nim=True, include_matkul_features=True):
    encoded_transformer_features = []
    encoded_other_features = []

    if include_nim:
        # Convert string input into integer indices
        nim_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY['nim']
        nim_lookup = StringLookup(vocabulary=nim_vocabulary, mask_token=None, num_oov_indices=0)(
            inputs['nim'])
        # Compute embedding dims
        embedding_dims = int(sqrt(len(nim_vocabulary)))
        # Create an embedding layer with specified dimensions
        embedding_encoder = Embedding(
            input_dim=len(nim_vocabulary),
            output_dim=embedding_dims,
            name=f'nim_embedding',
        )
        # Convert index values to embedding representations
        encoded_other_features.append(embedding_encoder(nim_lookup))

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
    positions = tf_range(start=0, limit=sequence_length - 1, delta=1)
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


def create_model():
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, include_nim, include_matkul_features)

    # go through encoder layer(s)
    transformer_features = encoder_layer(transformer_features)
    features = Flatten()(transformer_features)

    # concatenate with other features
    if other_features is not None:
        features = concatenate(
            [features, Reshape([other_features.shape[-1]])(other_features)])

    # mlp layers and loss function
    for num_units in hidden_units:
        features = Dense(num_units)(features)
        features = BatchNormalization()(features)
        features = LeakyReLU()(features)
        features = Dropout(dropout_rate)(features)

    outputs = Dense(units=1)(features)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def evaluate_model(dataset, sequence_length, step_size):
    path = fr'{main_path}/seq-{sequence_length}/step-{step_size}'
    Path(path).mkdir(parents=True, exist_ok=True)

    dataset['mk_nilai'] = dataset.apply(lambda x: create_sequences(
        (x['mk_nilai_1_6'], x['mk_nilai_7']), sequence_length, step_size), axis=1)

    del dataset['mk_nilai_1_6'], dataset['mk_nilai_7']

    random_selection = rand(len(dataset.index)) <= training_dataset_size
    train_data = dataset[random_selection]
    test_data = dataset[~random_selection]

    train_data = finalize_dataframe(train_data)
    test_data = finalize_dataframe(test_data)

    train_data.to_csv(f'{path}/train_data.csv',
                      index=False, sep='|', header=False)
    test_data.to_csv(f'{path}/test_data.csv',
                     index=False, sep='|', header=False)

    clear_session()
    model = create_model()
    plot_model(model, to_file=f'{path}/model.png', dpi=300, show_shapes=True, show_layer_activations=True,
               show_dtype=True, expand_nested=True)

    model.compile(
        optimizer=optimizer_function,
        loss=loss_function,
        metrics=metrics_function,
    )

    train_dataset = get_dataset_from_csv(
        f'{path}/train_data.csv', shuffle=True, batch_size=batch_size)

    start_time = time()
    history = model.fit(train_dataset, epochs=epochs)
    finish_time = time()
    with open(f'{path}/prediction-result.txt', 'w') as f:
        print(
            f'training time: {timedelta(seconds=(finish_time - start_time))}\n', file=f)

    # Create training graph
    plot(history.history['loss'])
    plot(history.history['rmse'])
    title('Training Result')
    ylabel('Value')
    xlabel('Epochs')
    legend(['Loss', 'RMSE'], loc='upper left')
    savefig(f'{path}/training_graph.png', dpi=300.0, format='png')

    # Test the model
    test_dataset = get_dataset_from_csv(
        f'{path}/test_data.csv', batch_size=batch_size)
    loss, rmse = model.evaluate(test_dataset, verbose=0)

    loss = round(loss, 5)
    rmse = round(rmse, 5)

    with open(f'{path}/prediction-result.txt', 'a') as f:
        print(f"Test Loss: {loss}", file=f)
        print(f"Test RMSE: {rmse}", file=f)

    # Save the model
    model.save(f'{path}/saved-model')

    return loss, rmse


sequences = []
losses = []
losses_labels = []
rmse_values = []
rmse_labels = []
steps_percentage = ['10', '25', '50', '75', '90']
steps_sizes = []
stats_records = []

# run the evaluations
evaluation_started = time()
for sequence_length in range(5, 61, 5):
    steps = []
    percentage = [(int(step) / 100) for step in steps_percentage]
    if sequence_length < 20:
        steps = [int(sequence_length * multiplier)
                 for multiplier in percentage[1:4]]
    else:
        steps = [int(sequence_length * multiplier)
                 for multiplier in percentage]

    losses_in_sequence = []
    rmse_values_in_sequence = []
    for step_size in steps:
        loss, rmse = evaluate_model(
            nilai_data.copy(), sequence_length, step_size)
        losses_in_sequence.append(loss)
        rmse_values_in_sequence.append(rmse)
        stats_records.append([f'{sequence_length}-{step_size}', loss, rmse])

    if sequence_length < 20:
        steps.insert(0, 'skip')
        steps.append('skip')
        losses_in_sequence.insert(0, 0)
        losses_in_sequence.append(0)
        rmse_values_in_sequence.insert(0, 0)
        rmse_values_in_sequence.append(0)

    sequences.append(str(sequence_length))
    steps_sizes.append(str(steps))
    losses.append(losses_in_sequence)
    losses_labels.append(['NaN' if item == 0 else str(item)
                         for item in losses_in_sequence])
    rmse_values.append(rmse_values_in_sequence)
    rmse_labels.append(
        ['NaN' if item == 0 else str(item) for item in rmse_values_in_sequence])
evaluation_finished = time()
system('cls')


def string_time(time):
    return strftime('%H:%M:%S %z', localtime(time))


stats_records.sort(key=itemgetter(1, 2))

# Save evaluation stats to file
with open(f'{main_path}/evaluation_stats.txt', 'w') as f:
    print(f'Date: {date_time.strftime(long_date_fmt)}', file=f)
    print(
        f'Evaluation started: {string_time(evaluation_started)}', file=f)
    print(f'Evaluation finished: {string_time(evaluation_finished)}', file=f)
    print(
        f'Duration: {timedelta(seconds=(evaluation_finished - evaluation_started))}\n', file=f)
    print(f'Epochs: {epochs}', file=f)
    print(f'Batch size: {batch_size}', file=f)
    print(f'Learning rate: {learning_rate}', file=f)
    print(f'Dropout rate: {dropout_rate}', file=f)
    print(f'Number of Attention Head(s): {num_heads}', file=f)
    print(f'Number of Transformer Layer: {num_layers}\n', file=f)
    print(
        f'Size of Training dataset: {int(training_dataset_size * 100)}%', file=f)
    print(
        f'Size of Testing dataset: {int(round(1 - training_dataset_size) * 100)}%', file=f)
    print(
        f'Sequence\'s fraction for count the steps: {steps_percentage}', file=f)
    print('Sequences and the combination of step sizes:', file=f)
    for i, sequence_length in enumerate(sequences):
        print(f'{sequence_length}: {steps_sizes[i]}', file=f)
    print()
    print(
        f'Best combination: {stats_records[0][0]} (Loss: {stats_records[0][1]}; RMSE: {stats_records[0][2]})', file=f)
    print(
        f'Worst combination: {stats_records[-1][0]} (Loss: {stats_records[-1][1]}; RMSE: {stats_records[-1][2]})', file=f)
    print(
        f'Average Loss: {average([record[1] for record in stats_records])}', file=f)
    print(
        f'Average RMSE: {average([record[2] for record in stats_records])}', file=f)


# Plot Creation


def split_list(list, num_split):
    splitted_list = []
    start = 0
    delta = len(list) // num_split

    while start < len(list):
        end = start + delta
        splitted_list.append(list[start:end])
        start = end

    return splitted_list


def split_and_transpose_list(l, num_split):
    return [list(map(list, zip(*x))) for x in split_list(l, num_split)]


def create_plot(xaxis, types, values, data_labels, value_labels, parts=None):
    positions = list(range((len(xaxis))))  # the label locations
    width = 0.19  # the width of the bars
    bars = []
    _, ax = subplots()

    for i, label in enumerate(data_labels):
        delta = 0.125 + width * i
        bar = ax.bar([pos + delta for pos in positions],
                     values[i], width, label=f'{label}%')
        bars.append(bar)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(positions)

    def update_ticks(_, pos):
        return xaxis[pos]

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(FuncFormatter(update_ticks))
    ax.xaxis.set_minor_locator(FixedLocator([p+0.5 for p in positions]))
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    xlim(min(positions), max(positions)+1)
    ylim([0, 0.5 + max([max(steps) for steps in values])])

    for idx, bar in enumerate(bars):
        ax.bar_label(
            bar, labels=value_labels[idx], padding=1, rotation='vertical')

    grid()
    xlabel('Sequences')
    ylabel(types)
    title(f'{types} by Sequences and Steps')
    legend()

    if parts is None:
        savefig(f'{main_path}/{types.lower()}.png', dpi=300.0, format='png')
    else:
        savefig(f'{main_path}/{types.lower()}-{parts}.png',
                dpi=300.0, format='png')


num_split = 3
sequences = split_list(sequences, num_split)
losses = split_and_transpose_list(losses, num_split)
losses_labels = split_and_transpose_list(losses_labels, num_split)
rmse_values = split_and_transpose_list(rmse_values, num_split)
rmse_labels = split_and_transpose_list(rmse_labels, num_split)

for part, (x, y, z) in enumerate(zip(sequences, losses, losses_labels)):
    create_plot(x, 'Losses', y, steps_percentage, z, part)

for part, (x, y, z) in enumerate(zip(sequences, rmse_values, rmse_labels)):
    create_plot(x, 'RMSE Values', y, steps_percentage, z, part)
