from flask import Flask, render_template, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from Recommendor import Recommender
from Models import db, User, Rating

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommender.db'
app.config['SECRET_KEY'] = 'joannaisthebest'
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)

recommender = Recommender()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    user_input = request.form['user_input']
    recommendations = recommender.get_recommendations(current_user.id)
    movie_titles = [recommender.movies[recommender.movies['movieId'] == movie_id]['title'].values[0] for movie_id in recommendations]
    return render_template('recommend.html', recommendations=movie_titles)

if __name__ == '__main__':
    app.run(debug=True)
