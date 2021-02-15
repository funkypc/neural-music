import glob
import pickle
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from midi2audio import FluidSynth
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding, BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from music21 import converter, instrument, note, chord, interval, pitch, key, midi, stream, environment, meter

model = None
nn_input = None
note_names = None
n_vocab = None

MODEL_DIR = 'g:/My Drive/MLData'
MODEL_DIR2 = 'c:/MLData'
GENERATED_DIR = 'C:/Users/User/projects/Python/Flask-App/static'

# Commented out IPython magic to ensure Python compatibility.
# !wget https://lilypond.org/download/binaries/linux-64/lilypond-2.22.0-1.linux-64.sh
# !sh lilypond-2.22.0-1.linux-64.sh

# Install dependencies
# !sudo apt-get install musescore
# !sudo apt-get install fluidsynth
# !pip install midi2audio

# Create the user environment for music21
# !whereis musescore
us = environment.UserSettings()
us['musicxmlPath'] = 'g:/My Drive/apps/MuseScore/MuseScorePortable.exe'
us['musescoreDirectPNGPath'] = 'g:/My Drive/apps/MuseScore/MuseScorePortable.exe'
us['graphicsPath'] = MODEL_DIR  # should be the application to open png file when using lilypond
us['lilypondPath'] = 'C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe'


# us['directoryScratch'] = 'drive/MyDrive/MLData/output' # Where to output temporary png files

# %env QT_QPA_PLATFORM=offscreen

def parse_midi():
    notes = np.array([])
    for filename in glob.glob(MODEL_DIR + "/midi_songs/*.mid"):
        stream = converter.parse(filename)
        sNew = change_key(stream)
        notes = np.append(notes, parse_notes(sNew))
    # write notes to file
    with open(MODEL_DIR + '/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)


# Converts all midi data in the same key
def change_key(stream, key_sig='C'):
    k = stream.analyze('key')
    i = interval.Interval(k.getScale('major').tonic, pitch.Pitch(key_sig))
    sNew = stream.transpose(i)
    return sNew


def parse_notes(stream):
    notes = np.array([])
    notes_to_parse = None
    ks = key.Key('C')  # set key signature
    try:  # stream has multiple instruments
        s2 = instrument.partitionByInstrument(stream)
        notes_to_parse = s2.parts[0].recurse()
    except:  # stream has flat structure
        notes_to_parse = stream.flat.notes
    for item in notes_to_parse:
        if isinstance(item, note.Note):
            # Fix incorrect notes # TODO plot
            nStep = item.pitch.step
            rightAccidental = ks.accidentalByStep(nStep)
            item.pitch.accidental = rightAccidental
            notes = np.append(notes, str(item.pitch))
    return notes


# def generate_song(model, nn_input, note_names, n_vocab):
def generate_song(transpose=0, time_sig=44, length=64):
    # Get notes from neural network
    notes = get_notes(model, nn_input, note_names, n_vocab, time_sig, length)
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
    # Generate Music Score using lilypond - not needed. Done in html and js
    # s1.write('lily.png', GENERATED_DIR + '/score')  # generates score.png
    # Transpose notes
    if transpose != 0:
        a_interval = interval.Interval(transpose)
        s1.transpose(a_interval, inPlace=True)
    # Create midi file from notes
    fp = s1.write('midi', fp=GENERATED_DIR + '/output.mid')
    # Convert midi file to audio using FluidSynth - Not needed. Done in html and js
    # fs = FluidSynth('c:\MLData\Yamaha-Grand-Lite-v2.0.sf2')
    # fs.play_midi(GENERATED_DIR + '/output.mid')
    # fs.midi_to_audio(GENERATED_DIR + '/output.mid', GENERATED_DIR + '/output.wav')


def get_notes(model, nn_input, note_names, n_vocab, time_sig, length):
    # choose a random starting note
    # random_start = np.random.randint(0, len(nn_input) - 1)
    random_start = np.random.randint(0, 200)
    sequence = nn_input[random_start]
    output = []
    # generate notes
    for i in range(length):
        inputs = np.reshape(sequence, (1, len(sequence), 1))
        inputs = inputs / float(n_vocab)
        prediction = model.predict(inputs, verbose=0)
        index = np.argmax(prediction)
        res = note_names[index]
        output.append(res)
        sequence = np.append(sequence, index)
        sequence = sequence[1:len(sequence)]
    return output


def get_plot():
    s1 = stream.Stream()
    for filename in glob.glob(MODEL_DIR + "/midi_songs/*.mid"):
        s1.append(converter.parse(filename))
    return s1.plot('scatter', 'offset', 'pitchClass')



def create_model(nn_input, n_vocab):
    # Generate notes using neural network
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

    return model1


def train_network(nn_input, nn_output, model):
    # One-hot encode nn_output
    nn_output = np_utils.to_categorical(nn_output)
    # train the network
    weightsfile = MODEL_DIR + '/weights/weights-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(weightsfile, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # Load history
    # history = pickle.load(open('/trainHistoryDict', "rb"))
    # Create history
    history = model.fit(nn_input, nn_output, epochs=200, batch_size=100, callbacks=callbacks_list)
    # Save history to plot from later
    with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Uncomment to plot history
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()


def init_network():
    global model
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

    # Load Weights
    model.load_weights(
        MODEL_DIR + '/weights/weights-136-0.1883.hdf5')  # Todo: try weights-82-0.4268 weights-136-0.1883.hdf5

    # Train the network
    # train_network(nn_input, nn_output, model) # Uncomment to retrain the network


if __name__ == "__main__":
    init_network()
    generate_song()
