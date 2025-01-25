import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


# Charger les données
data = pd.read_csv("data_filtered (3).csv")  # Remplacez avec le chemin réel de votre fichier

# Colonnes utilisées pour les fonctionnalités
features = ['Goals', 'Assists', 'Shots on target', 'Big chances created', 'Passes', 'Tackles']

# Préparer les données pour les recommandations
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Modèle de recherche des voisins les plus proches
neighbor_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
neighbor_model.fit(data_scaled)



# Initialiser l'application Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = '748cb5504a932d6d92bd60ac916476fe'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/flask_app_db'  # MySQL connection
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize the database
with app.app_context():
    db.create_all()
@app.route('/', methods=['GET', 'POST'])
def login():
    print("Login route triggered")

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('accueil'))
        else:
            flash('Login failed. Check email and password.', 'danger')
    return render_template('login.html')

# Route pour la page d'accueil
@app.route('/accueil')

def accueil():
    return render_template('accueil.html')

# Route pour la page d'index
@app.route('/index')
def index():
    return render_template('index.html')

# Route pour la recommandation des 5 meilleurs joueurs
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    positions = data['Position'].unique()  # Récupérer toutes les positions uniques
    top_players = []

    # Vérifier si une position a été sélectionnée
    if request.method == 'POST':
        selected_position = request.form.get('position')
        # Filtrer les joueurs par position et prendre les 5 meilleurs
        top_players = data[data['Position'] == selected_position].nlargest(5, 'Performance Score')[
            ['Name', 'Club', 'Position', 'Performance Score']].to_dict(orient='records')

    return render_template(
        'recommend.html',
        positions=positions,
        top_players=top_players
    )
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# Route pour le traitement et l'affichage des résultats
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    input_features = [
        float(request.form['goals']),
        float(request.form['assists']),
        float(request.form['shots_on_target']),
        float(request.form['big_chances_created']),
        float(request.form['passes']),
        float(request.form['tackles'])
    ]

    # Standardiser les données d'entrée
    input_data = scaler.transform([input_features])

    # Trouver les joueurs similaires
    distances, indices = neighbor_model.kneighbors(input_data)
    recommended_players = data.iloc[indices[0]]

    # Prédire une position
    predicted_position = recommended_players['Position'].mode()[0]

    # Retourner les résultats
    return render_template(
        'result.html',
        position=predicted_position,
        recommendations=recommended_players[['Name', 'Club', 'Position', 'Goals', 'Assists']].to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)

