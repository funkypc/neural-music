import base64
import random
import nn
import numpy as np
import pickle
from io import BytesIO
from flask import Flask, request, render_template, flash, redirect, send_from_directory
from flask_login import LoginManager, login_user, UserMixin, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from matplotlib.figure import Figure

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "/login"
app.secret_key = b'_6#e4F"S2Z8z\n\xec]/'
song = None


# Index - Authenticated users only
@app.route('/')
@login_required
def index():
    return render_template("index.html", generate_song=generate_song, song=song)


# Visualization - Authenticated users only
@app.route('/visualization')
@login_required
def visualization():
    return render_template("visualization.html", get_figure=get_figure, get_model_summary=get_model_summary)


# Login
@app.route('/login')
def login():
    form = LoginForm()
    return render_template("login.html", form=form)


# Login POST method
@app.route('/login', methods=['POST'])
def login_post():
    global user
    # recreate default user - bugfix
    user = User()
    username = request.form.get('username')
    password = request.form.get('password')
    if username == user.username and password == user.password:
        login_user(user)
        return redirect("/")
    else:
        flash("Unable to log in")
        return redirect("/login")


# Check if user is authenticated
@login_manager.user_loader
def load_user(user_id):
    global user
    user = User()  # bugfix user
    if int(user_id) == user.id:
        return user
    # Return None of user is not authenticated
    return None


# Log out
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")


# Generate song
@app.route('/generate_song')
@app.route('/generate_song/<path:transpose>/<path:time_sig>/<path:length>')
@login_required
def generate_song(transpose=0, time_sig=44, length=32):
    if request.args.get('length', ''):
        length = request.args.get('length', '')
    if request.args.get('time_sig', ''):
        time_sig = request.args.get('time_sig', '')
    if request.args.get('transpose', ''):
        transpose = request.args.get('transpose', '')
    nn.generate_song(int(transpose), int(time_sig), int(length))
    global song
    song = True
    return redirect("/")


# Get file without using cache
@app.route('/media/<path:filename>')
@login_required
def get_media(filename):
    return send_from_directory('static/', filename, cache_timeout=0)


# Get a matplotlib figure for visualization. Embed in html.
def get_figure(name):
    # Generate figure
    if name is "note_frequency":
        fig = Figure(figsize=(14, 5))
        fig.suptitle("Note Frequency in Key of C", fontsize=18)
        ax = fig.subplots()
        with open(nn.MODEL_DIR + '/notes', 'rb') as filepath:
            notes = np.array(pickle.load(filepath))
        unique, counts = np.unique(notes, return_counts=True)
        ax.bar(unique, counts)
    elif name is "key_frequency":
        fig = Figure()
        fig.suptitle("Key Signature Frequency")
        ax = fig.subplots()
        ax.plot([1, 2])
    elif name is "midi_data":
        # Generate Multiple Correspondence Analysis (MCA) graph
        X, mca = nn.generate_mca_graph()
        ax = mca.plot_coordinates(
            X=X,
            ax=None,
            figsize=(20, 8),
            show_row_points=True,
            row_points_size=10,
            show_row_labels=False,
            show_column_points=True,
            column_points_size=30,
            show_column_labels=True,
            legend_n_cols=1
        )
        fig = ax.get_figure()
    else:
        fig = Figure()
        fig.suptitle("Generic Title")
        ax = fig.subplots()
        ax.plot([1, 2])

    # Save to temporary buffer
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed result in html output
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}' style='max-width:100%'/>"


# Get summary of Neural Network model to display to the user
def get_model_summary():
    # summary = []
    # # create model
    # model = nn.create_model(nn.nn_input, nn.n_vocab)
    # model.summary(print_fn=lambda x: summary.append(x))
    # res = ''
    # for line in summary:
    #     res += line + '<br>'
    # # Return hard-coded model details to save memory
    res = 'Model: "sequential"<br>_________________________________________________________________<br>Layer (type)                 Output Shape              Param #   <br>=================================================================<br>lstm (LSTM)                  (None, 8, 512)            1052672   <br>_________________________________________________________________<br>lstm_1 (LSTM)                (None, 8, 512)            2099200   <br>_________________________________________________________________<br>lstm_2 (LSTM)                (None, 512)               2099200   <br>_________________________________________________________________<br>batch_normalization (BatchNo (None, 512)               2048      <br>_________________________________________________________________<br>dropout (Dropout)            (None, 512)               0         <br>_________________________________________________________________<br>dense (Dense)                (None, 256)               131328    <br>_________________________________________________________________<br>activation (Activation)      (None, 256)               0         <br>_________________________________________________________________<br>batch_normalization_1 (Batch (None, 256)               1024      <br>_________________________________________________________________<br>dropout_1 (Dropout)          (None, 256)               0         <br>_________________________________________________________________<br>dense_1 (Dense)              (None, 48)                12336     <br>_________________________________________________________________<br>activation_1 (Activation)    (None, 48)                0         <br>=================================================================<br>Total params: 5,397,808<br>Trainable params: 5,396,272<br>Non-trainable params: 1,536<br>_________________________________________________________________<br>'
    return res


# Class to create login form
class LoginForm(FlaskForm):
    username = StringField('', [DataRequired()])
    password = PasswordField('', [DataRequired()])
    submit = SubmitField('Sign in')


# Class to create default user
class User(UserMixin):
    username = "user"
    password = "user"
    random.seed(27)
    id = random.randint(1000000, 9999999)  # Generate unique ID number at app startup


# init app
if __name__ == "__main__":
    # create default user
    user = User()
    # Run app
    app.run(debug=True)
