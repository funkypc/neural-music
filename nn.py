import gc
import glob
import pickle
import pandas as pd
import numpy as np
import prince
# import matplotlib.pyplot as plt
import tensorflow as tf
# from midi2audio import FluidSynth
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding, BatchNormalization as BatchNorm
from tensorflow.keras.utils import plot_model
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from music21 import converter, instrument, note, chord, interval, pitch, key, midi, stream, environment, meter, bar

# MODEL_DIR = 'g:/My Drive/MLData'
# GENERATED_DIR = 'C:/Users/User/projects/Python/Flask-App/static'
MODEL_DIR = './static'
GENERATED_DIR = './static'


nn_input = pickle.load(open(MODEL_DIR + '/nn_input.pkl', "rb"))
note_names = pickle.load(open(MODEL_DIR + '/note_names.pkl', "rb"))
n_vocab = pickle.load(open(MODEL_DIR + '/n_vocab.pkl', "rb"))


# Parse the midi files and dump all notes to file
def parse_midi():
    notes = np.array([])
    for filename in glob.glob(MODEL_DIR + "/midi_songs/*.mid"):
        stream1 = converter.parse(filename)
        # transpose midi stream to key of C
        stream1 = change_key(stream1)
        # fix notes not in key of C
        notes_array = parse_notes(stream1)
        # append notes to array
        notes = np.append(notes, notes_array)
    # write notes to file
    with open(MODEL_DIR + '/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)


# Transpose all songs (streams) into key of C
def change_key(stream1, key_sig='C'):
    k = stream1.analyze('key')
    i = interval.Interval(k.getScale('major').tonic, pitch.Pitch(key_sig))
    stream1 = stream1.transpose(i)
    return stream1


def parse_notes(stream1):
    notes = np.array([])
    ks = key.Key('C')  # set key signature to C major
    try:  # if stream has multiple instruments
        s2 = instrument.partitionByInstrument(stream1)
        notes_to_parse = s2.parts[0].recurse()
    except:  # else stream has flat structure
        notes_to_parse = stream1.flat.notes
    for item in notes_to_parse:
        if isinstance(item, note.Note):
            # Bump notes not in key signature to closest note in key signature
            n_step = item.pitch.step
            right_accidental = ks.accidentalByStep(n_step)
            item.pitch.accidental = right_accidental
            notes = np.append(notes, str(item.pitch))
    return notes


# Generate a song using LSTM RNN
def generate_song(transpose=0, time_sig=44, length=64):
    # Get notes from neural network
    # create model
    model = create_model(nn_input, n_vocab)
    # Get sequence of notes from Neural Network
    notes = get_notes(model, nn_input, note_names, n_vocab, length)
    s1 = stream.Stream()
    # Set Time Signature
    ts = meter.TimeSignature('4/4')
    if time_sig == 24:
        ts = meter.TimeSignature('2/4')
    elif time_sig == 34:
        ts = meter.TimeSignature('3/4')
    s1.insert(0, ts)
    # Write notes to MIDI file
    for n in notes:
        # Append all notes as quarter notes
        s1.append(note.Note(n, type='quarter'))
    # Generate music score using lilypond - Unused. Currently score image is created with html-midi-player in index.html
    # s1.write('lily.png', GENERATED_DIR + '/score')  # generates score.png
    # Transpose song
    if transpose != 0:
        a_interval = interval.Interval(transpose)
        s1.transpose(a_interval, inPlace=True)
    # Create midi file from notes
    fp = s1.write('midi', fp=GENERATED_DIR + '/output.mid')
    # Convert midi file to audio using FluidSynth - Unused. Done in html with html-midi-player.
    # fs = FluidSynth('c:\MLData\Yamaha-Grand-Lite-v2.0.sf2')
    # fs.midi_to_audio(GENERATED_DIR + '/output.mid', GENERATED_DIR + '/output.wav')
    # Cleanup model
    del model
    gc.collect()


# Get sequence of notes from Neural Network
def get_notes(model, nn_input, note_names, n_vocab, length):
    # choose a random starting note
    random_start = np.random.randint(0, len(nn_input) - 1)
    sequence = nn_input[random_start]
    output = []
    # generate notes
    for i in range(length+16):
        inputs = np.reshape(sequence, (1, len(sequence), 1))
        inputs = inputs / float(n_vocab)
        # Predict next note
        prediction = model.predict(inputs, verbose=0)
        index = np.argmax(prediction)
        res = note_names[index]
        output.append(res)
        sequence = np.append(sequence, index)
        sequence = sequence[1:len(sequence)]
    random_start = np.random.randint(0, len(output) - length - 1)
    random_end = random_start + length
    output = output[random_start:random_end]
    return output


# Generate Multiple Correspondence Analysis (MCA) graph for data visualization
def generate_mca_graph():
    df = get_stream_df()
    X = df
    mca = prince.MCA()
    mca = mca.fit(X)
    return X, mca


def get_stream_df():
    # Recreate dataframe
    s1 = stream.Stream()
    for filename in glob.glob(MODEL_DIR + "/midi_songs/*.mid"):
        s1.append(converter.parse(filename))
    df = generate_dataframe(s1)
    df.to_pickle(MODEL_DIR + "/midi_df.pkl")

    # Read DataFrame from file
    df = pd.read_pickle(MODEL_DIR + "/midi_df.pkl")
    # Drop NaN columns
    df = df.dropna()
    # Convert fields to numerical
    df = df.astype({'Offset': 'float'})
    df = df.astype({'Duration': 'float'})
    # Cleanup outliers in Duration. Keep to 3 standard deviations
    df = df[np.abs(df.Duration - df.Duration.mean()) <= (3 * df.Duration.std())]
    return df


# Generate pandas dataFrame from stream
def generate_dataframe(stream1):
    rows_list = []
    for index, elt in enumerate(stream1.flat.stripTies(retainContainers=True).getElementsByClass(
            [note.Note, note.Rest, chord.Chord, bar.Barline])):
        if hasattr(elt, 'pitches'):
            pitches = elt.pitches
            for pitch in pitches:
                rows_list.append(generate_row(elt, stream1, pitch.pitchClass))
            else:
                rows_list.append(generate_row(elt, stream1))
    return pd.DataFrame(rows_list)


# Generate row in dataframe
def generate_row(mus_object, stream1, pitch_class=np.nan):
    d = {}
    try:
        pitch = mus_object.pitch
    except:
        pitch = None
    d.update({'id': mus_object.id,
              'Offset': mus_object.offset,
              'Duration': mus_object.duration.quarterLength,
              'Pitch': pitch,
              'Pitch Class': pitch_class})
    return d


# Create LSTM Recurrent Neural Network model
def create_model(nn_input, n_vocab):
    # Generate notes using Recurring Neural Network (RNN)
    # Create model
    model1 = Sequential()
    model1.add(LSTM(
        512, input_shape=(nn_input.shape[1], nn_input.shape[2]),
        recurrent_dropout=0.2, return_sequences=True
    ))
    model1.add(LSTM(512, return_sequences=True, recurrent_dropout=0.1, ))
    model1.add(LSTM(512))
    model1.add(BatchNorm())
    model1.add(Dropout(0.1))
    model1.add(Dense(256))
    model1.add(Activation('relu'))
    model1.add(BatchNorm())
    model1.add(Dropout(0.4))
    model1.add(Dense(n_vocab))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # Load Weights
    model1.load_weights(MODEL_DIR + '/weights/weights-136-0.1883.hdf5')

    return model1


# Train the Neural Network
def train_network(nn_input, nn_output, model):
    # OneHotEncode nn_output
    nn_output = np_utils.to_categorical(nn_output)
    # train the network
    weights_file = MODEL_DIR + '/weights/weights-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(weights_file, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # Load history
    # history = pickle.load(open(MODEL_DIR + '/trainHistoryDict', "rb"))
    # Create history with 200 epochs
    history = model.fit(nn_input, nn_output, epochs=200, batch_size=100, callbacks=callbacks_list)
    # Save history to file
    with open(MODEL_DIR + '/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Plot history
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()


# Initialize the network.
def init_network():
    global nn_input
    global note_names
    global n_vocab
    with open(MODEL_DIR + '/notes', 'rb') as filepath:
        notes = np.array(pickle.load(filepath))
    # set length of sequence
    sq_length = np.random.randint(5, 10)
    # get amount of unique notes
    n_vocab = len(set(notes))
    # get sorted note names
    note_names = sorted(set(note for note in notes))
    # map note names to int using hashtable.
    note_hash = dict((note, number) for number, note in enumerate(note_names))
    # input and output note sequences
    nn_input = []
    nn_output = []

    # create input and output sequences
    for i in range(0, len(notes) - sq_length, 1):
        sq_in = notes[i:i + sq_length]
        sq_out = notes[i + sq_length]
        nn_input.append([note_hash[note] for note in sq_in])
        nn_output.append(note_hash[sq_out])

    # format input for compatibility with LSTM layers
    nn_input = np.reshape(nn_input, (len(nn_input), sq_length, 1))
    nn_input = nn_input / float(n_vocab)

    # create model
    model = create_model(nn_input, n_vocab)

    # Plot the model
    plot_model(model, to_file=GENERATED_DIR + '/model_plot.png', show_shapes=True, show_layer_names=True)

    # Train the network
    train_network(nn_input, nn_output, model)

    # convert public variables to pickle
    with open(MODEL_DIR + '/nn_input.pkl', 'wb') as filepath:
        pickle.dump(nn_input, filepath)
    with open(MODEL_DIR + '/note_names.pkl', 'wb') as filepath:
        pickle.dump(note_names, filepath)
    with open(MODEL_DIR + '/n_vocab.pkl', 'wb') as filepath:
        pickle.dump(n_vocab, filepath)


# init network and retrain model
if __name__ == "__main__":
    init_network()
    generate_song()
