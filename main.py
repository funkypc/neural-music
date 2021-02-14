import random
from flask import Flask, request, render_template, flash, redirect
from flask_login import LoginManager, login_user, UserMixin, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "/login"
app.secret_key = b'_6#e4F"S2Z8z\n\xec]/'


# Index
@app.route('/')
@login_required
def index():
    return render_template("index.html")


# Info
@app.route('/info')
@login_required
def info():
    return "info"


# Login
@app.route('/login')
def login():
    form = LoginForm()
    return render_template("login.html", form=form)


@app.route('/login', methods=['POST'])
def login_post():
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


# Class to create login form
class LoginForm(FlaskForm):
    username = StringField(
        'Username',
        [DataRequired()]
    )
    password = PasswordField(
        'Password ',
        [DataRequired()]
    )
    submit = SubmitField('Submit')


# Class to create default user
class User(UserMixin):
    username = "user"
    password = "user"
    id = random.randint(1000000, 9999999)  # Generate unique ID number at app startup


# init app
if __name__ == "__main__":
    # create default user
    user = User()
    # Run app
    app.run(debug=True)
