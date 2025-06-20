# from flask import Flask, render_template, request, send_file
# import os
# import matplotlib

# from modules.elevation import add_elevation_to_csv
# from modules.enveloppe_conforme import init_regles, main_enveloppe_conforme
# from modules.generation_ifc import main_extrusion
# matplotlib.use('Agg')  # Utilise un backend sans interface GUI
# import matplotlib.pyplot as plt
# from werkzeug.utils import secure_filename
# from modules.conversion import ajouter_entete_csv, convert_coordinates_from_csv
# # from modules.traitement import extraire_points_emprise
# from modules.parcelle import extraire_points_emprise, visualiser_parcelle
# from modules.traitement import enregistrer_vertices_csv
# from matplotlib.patches import Polygon
# from scipy.spatial import ConvexHull
# import numpy as np
# import pandas as pd
# import random
# from matplotlib import cm
# import matplotlib.colors as mcolors

# from auth import db, login_manager, User
# from flask_login import current_user

# app = Flask(__name__)

# app.config['SECRET_KEY'] = 'pass_cle'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db.init_app(app)
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# with app.app_context():
#     db.create_all()

# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# file_ref = None  # variable globale
# def init_file_ref(file_path):
#     global file_ref
#     file_ref = file_path


# def dessiner_emprise(df_points):
#     points = df_points[["X", "Y"]].values
#     hull = ConvexHull(points)
#     vertices = points[hull.vertices]

#     fig, ax = plt.subplots(figsize=(12, 12))
#     plt.rcParams["font.family"] = "DejaVu Sans"
#     n_segments = len(vertices)
#     # colors = plt.colormaps.get_cmap('tab20', n_segments)
#     colors = cm.get_cmap('tab20', n_segments)

#     segments_info = []

#     for i in range(n_segments):
#         p1 = vertices[i]
#         p2 = vertices[(i + 1) % n_segments]
#         color = colors(i)

#         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3)
#         mid_x = (p1[0] + p2[0]) / 2
#         mid_y = (p1[1] + p2[1]) / 2
#         ax.text(mid_x, mid_y, str(i), fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))

#         segments_info.append({
#             "id": i,
#             "start": {"x": round(float(p1[0]), 2), "y": round(float(p1[1]), 2)},
#             "end": {"x": round(float(p2[0]), 2), "y": round(float(p2[1]), 2)},
#         })

#     ax.scatter(points[:, 0], points[:, 1], color='black', s=30)
#     ax.set_xlim(min(points[:, 0]) - 1, max(points[:, 0]) + 1)
#     ax.set_ylim(min(points[:, 1]) - 1, max(points[:, 1]) + 1)
#     # ax.set_title("Polygone 2D de l'emprise avec segments")
#     ax.set_aspect('equal')
#     ax.grid(True)

#     plt.savefig("static/emprise_polygone.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print(vertices)
#     enregistrer_vertices_csv(vertices)
#     ajouter_entete_csv()
#     add_elevation_to_csv()
#     return segments_info


# # def traiter_fichier_ifc(filepath):

# #     # Étape 1 : Récuperations des points du polygone de la parcelle 
# #     df_points = extraire_points_emprise(filepath)
# #     if df_points is None:
# #         return {"error": "Aucun élément avec Category=EMPRISE trouvé."}

# #     segments_info = dessiner_emprise(df_points)

# #     return {
# #         "elements": len(df_points),
# #         "sommets": len(df_points),
# #         "image_path": "static/emprise_polygone.png",
# #         "segments" : segments_info
# #     }



# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     resultats = None
# #     filename = ""

# #     if request.method == "POST":
# #         fichier = request.files.get("ifc-file")
# #         if fichier:
# #             print("obtenu")
# #             filename = secure_filename(fichier.filename)
# #             chemin = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #             fichier.save(chemin)
            
# #             resultats = traiter_fichier_ifc(chemin)

# #     return render_template("index.html", filename=filename, resultats=resultats)

# def segment_format(data):
#     segments = list(data.values())[0]  # Extraire la liste de segments

#     # Créer un dictionnaire pour retrouver les segments par leur point de départ
#     start_map = {
#         (seg['start']['x'], seg['start']['y']): seg
#         for seg in segments
#     }

#     # On commence par un segment arbitraire
#     first_seg = segments[0]
#     current_point = (first_seg['end']['x'], first_seg['end']['y'])

#     ordered = [first_seg]
#     used = {id(first_seg)}  # Pour éviter les doublons

#     # Itérer jusqu'à revenir au point de départ ou ne plus trouver de chaînage
#     while True:
#         next_seg = start_map.get(current_point)
#         if not next_seg or id(next_seg) in used:
#             break
#         ordered.append(next_seg)
#         used.add(id(next_seg))
#         current_point = (next_seg['end']['x'], next_seg['end']['y'])

#     # Vérifier si on revient au point de départ pour fermer la boucle
#     if ordered[0]['start'] != ordered[-1]['end']:
#         print("⚠️ La boucle n'est pas fermée.")
    
#     # Réaffecter les IDs
#     for i, seg in enumerate(ordered):
#         seg['id'] = i

#     return ordered

# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     resultats = None
# #     segments_info = []

# #     if request.method == "POST":
# #         f = request.files.get("ifc-file")
# #         if f:
# #             file_path = os.path.join("uploads", secure_filename(f.filename))
# #             f.save(file_path)

# #             init_file_ref(file_path)

# #             df_points = extraire_points_emprise(file_path)
# #             if df_points is None or df_points.empty:
# #                  return render_template("erreur.html")
# #             df_points = df_points.drop(columns=["Z"])

# #             # if(df_points.empty()):
# #             #     return render_template("erreur.html")
# #             # segments_info = dessiner_emprise(df_points)  # retourne les segments
# #             segments_info = visualiser_parcelle() 
# #             segments_info = segment_format(segments_info)
# #             print(segments_info)
# #             resultats = True
# #         #get pour recuperer si c'est fait par dessin

# #     return render_template("index.html", resultats=resultats, segments=segments_info)


# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# import os

# ALLOWED_EXTENSIONS = {'ifc'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route("/", methods=["GET", "POST"])
# def index():
#     resultats = None
#     segments_info = []

#     if request.method == "POST":
#         f = request.files.get("ifc-file")
#         if not f or not allowed_file(f.filename):
#             return render_template("erreur.html", message="Le fichier n'est pas un fichier IFC valide.")

#         file_path = os.path.join("uploads", secure_filename(f.filename))
#         f.save(file_path)

#         try:
#             init_file_ref(file_path)
#             df_points = extraire_points_emprise(file_path)
#             if df_points is None or df_points.empty:
#                 return render_template("erreur.html", message="Le fichier ne contient pas de données exploitables.")
#             df_points = df_points.drop(columns=["Z"])
#             segments_info = visualiser_parcelle() 
#             segments_info = segment_format(segments_info)
#             print(segments_info)
#             resultats = True
#         except Exception as e:
#             print("Erreur lors du traitement du fichier :", e)
#             return render_template("erreur.html", message="Une erreur est survenue lors du traitement du fichier.")

#     return render_template("index.html", resultats=resultats, segments=segments_info)



# @app.route("/traitement", methods=["POST"])
# def traiter_types():
#     building_height = float(request.form['building-height'])
#     init_regles(building_height)

#     segments_data = {}

#     for key, value in request.form.items():
#         if key.startswith("type-"):
#             seg_id = key.split("-")[1]
#             segments_data[seg_id] = {"type": value}
#         elif key.startswith("vue-"):
#             seg_id = key.split("-")[1]
#             if seg_id in segments_data:
#                 segments_data[seg_id]["vue"] = value

#     # Debug : afficher dans la console
#     print("Segments reçus :")
#     #for seg_id, data in segments_data.items():
#         #print(f"Segment {seg_id} -> Type: {data.get('type')}, Vue: {data.get('vue', 'N/A')}")
#     #print(segments_data)
    
#     type = [segments_data[key]['type'] for key in sorted(segments_data, key=int)]
#     vue = [segments_data[key]['vue'].lower() if segments_data[key]['vue'] else None for key in sorted(segments_data, key=int)]
#     main_enveloppe_conforme(type, vue)
#     return render_template("resultat.html", image_3d = "static/enveloppe_3d.png", image_enveloppe = "static/enveloppe_conforme.png")


# @app.route('/download')
# def download_file():
#     try:
#         # Chemin du fichier IFC généré
#         # file_reference = filepath
#         # print(file_path)
#         file_to_download = main_extrusion(file_ref)
#         #filepath = os.path.join("outputs", 'parcelle.ifc') 
#         return send_file(file_to_download, as_attachment=True)
#     except Exception as e:
#         return str(e), 404


# @app.route("/test")
# def test():
#     return render_template("test.html")

# @app.route("/viewer")
# def viewer():
#     return render_template("viewer.html")

















# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
        
#         if User.query.filter_by(email=email).first():
#             flash('Email déjà utilisé')
#             return redirect(url_for('register'))

#         new_user = User(email=email)
#         new_user.set_password(password)
#         db.session.add(new_user)
#         db.session.commit()

#         login_user(new_user)
#         return redirect(url_for('index'))
#     return render_template('register.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         user = User.query.filter_by(email=email).first()

#         if user and user.check_password(password):
#             login_user(user)
#             return redirect(url_for('index'))
        
#         flash('Email ou mot de passe incorrect')
#     return render_template('login.html')


# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('index'))







# if __name__ == '__main__':
#     app.run(debug=True)



##########################################################################
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import os
import matplotlib
import datetime
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from modules.elevation import add_elevation_to_csv
from modules.enveloppe_conforme import init_regles, main_enveloppe_conforme
from modules.generation_ifc import main_extrusion
from modules.conversion import ajouter_entete_csv, convert_coordinates_from_csv
from modules.parcelle import extraire_points_emprise, visualiser_parcelle
from modules.traitement import enregistrer_vertices_csv
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.colors as mcolors
from models import db, User, Analysis  # Import depuis models.py
from flask import session
from dotenv import load_dotenv
load_dotenv()



matplotlib.use('Agg')  # Backend non interactif pour matplotlib

# # Initialisation de l'application Flask
# app = Flask(__name__)

# # Configuration de l'application
# app.config['SECRET_KEY'] = 'cledetest'  # À changer en production
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'ifc'}

# # Initialisation des extensions
# login_manager = LoginManager(app)
# login_manager.login_view = 'login'



# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))


# # Création des tables de la base de données
# with app.app_context():
#     db.create_all()


# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration de l'application
app.config.update({
    'SECRET_KEY': os.getenv('SECRET_KEY'),#'cledetest',  
    'SQLALCHEMY_DATABASE_URI': os.getenv('DATABASE_URL'),#'sqlite:///database.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'UPLOAD_FOLDER': 'uploads',
    'ALLOWED_EXTENSIONS': {'ifc'}
})

# Initialisation des extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configuration des dossiers
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# # Création des tables de la base de données
with app.app_context():
    db.create_all()

# Variable globale pour le fichier de référence
file_ref = None

def init_file_ref(file_path):
    global file_ref
    file_ref = file_path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fonctions utilitaires
def dessiner_emprise(df_points):
    points = df_points[["X", "Y"]].values
    hull = ConvexHull(points)
    vertices = points[hull.vertices]

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams["font.family"] = "DejaVu Sans"
    n_segments = len(vertices)
    colors = cm.get_cmap('tab20', n_segments)

    segments_info = []
    for i in range(n_segments):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_segments]
        color = colors(i)

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, str(i), fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))

        segments_info.append({
            "id": i,
            "start": {"x": round(float(p1[0]), 2), "y": round(float(p1[1]), 2)},
            "end": {"x": round(float(p2[0]), 2), "y": round(float(p2[1]), 2)},
        })

    ax.scatter(points[:, 0], points[:, 1], color='black', s=30)
    ax.set_xlim(min(points[:, 0]) - 1, max(points[:, 0]) + 1)
    ax.set_ylim(min(points[:, 1]) - 1, max(points[:, 1]) + 1)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.savefig("static/emprise_polygone.png", dpi=300, bbox_inches='tight')
    plt.close()
    enregistrer_vertices_csv(vertices)
    ajouter_entete_csv()
    add_elevation_to_csv()
    return segments_info

def segment_format(data):
    segments = list(data.values())[0]
    start_map = {(seg['start']['x'], seg['start']['y']): seg for seg in segments}
    ordered = [segments[0]]
    used = {id(segments[0])}
    current_point = (segments[0]['end']['x'], segments[0]['end']['y'])

    while True:
        next_seg = start_map.get(current_point)
        if not next_seg or id(next_seg) in used:
            break
        ordered.append(next_seg)
        used.add(id(next_seg))
        current_point = (next_seg['end']['x'], next_seg['end']['y'])

    if ordered[0]['start'] != ordered[-1]['end']:
        print("⚠️ La boucle n'est pas fermée.")
    
    for i, seg in enumerate(ordered):
        seg['id'] = i

    return ordered

# Routes d'authentification
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Les mots de passe ne correspondent pas')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Cet email est déjà utilisé')
            return redirect(url_for('register'))

        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        #   flash('Inscription réussie!', 'success')
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        
        flash('Email ou mot de passe incorrect')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Routes principales
# @app.route("/", methods=["GET", "POST"])
# @login_required
# def index():
#     resultats = None
#     segments_info = []

#     if request.method == "POST":
#         f = request.files.get("ifc-file")
#         if not f or not allowed_file(f.filename):
#             #flash("Le fichier n'est pas un fichier IFC valide.", 'error')
#             # return redirect(url_for('index'))
#             return render_template("erreur.html")

#         try:
#             os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
#             f.save(file_path)

#             init_file_ref(file_path)

#             df_points = extraire_points_emprise(file_path)
            
#             if df_points is None or df_points.empty:
#                 # flash("Le fichier ne contient pas de données exploitables.", 'error')
#                 return render_template("erreur.html")
#                 # return redirect(url_for('index'))

#             df_points = df_points.drop(columns=["Z"])
#             segments_info = visualiser_parcelle() 
#             segments_info = segment_format(segments_info)
#             resultats = True
#             #flash("Fichier analysé avec succès!", 'success')
            
#         except Exception as e:
#             print(f"Erreur lors du traitement: {str(e)}")
#             #flash("Une erreur est survenue lors du traitement du fichier.", 'error')
#             # return redirect(url_for('index'))
#             return render_template("erreur.html")

#     return render_template("index.html", resultats=resultats, segments=segments_info)
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        f = request.files.get("ifc-file")
        if not f or not allowed_file(f.filename):
            return render_template("erreur.html")

        try:
            # Création du dossier upload si besoin
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(file_path)

            init_file_ref(file_path)

            # Création de l'analyse en base
            new_analysis = Analysis(
                user_id=current_user.id,
                # original_filename=f.filename,
                # status='uploaded'
            )
            db.session.add(new_analysis)
            db.session.commit()

            # Traitement avec l'ID de l'analyse
            df_points = extraire_points_emprise(file_path, new_analysis.id)  # Modification ici
            
            if df_points is None:
                return render_template("erreur.html")

            # Suite du traitement...
            df_points = df_points.drop(columns=["Z"])
            segments_info = visualiser_parcelle(new_analysis.id) 
            segments_info = segment_format(segments_info)

            session['analysis_id'] = new_analysis.id
            return render_template("index.html", resultats=True, segments=segments_info, analysis_id=new_analysis.id)

        except Exception as e:
            print(f"Erreur: {str(e)}")
            db.session.rollback()
            return render_template("erreur.html")

    return render_template("index.html")

@app.route("/traitement", methods=["POST"])
@login_required
def traiter_types():
    analysis_id = request.form.get("analysis_id")
    if not analysis_id:
        return render_template("erreur.html")
    try:
        building_height = float(request.form['building-height'])
        init_regles(building_height)

        segments_data = {}
        for key, value in request.form.items():
            if key.startswith("type-"):
                seg_id = key.split("-")[1]
                segments_data[seg_id] = {"type": value}
            elif key.startswith("vue-"):
                seg_id = key.split("-")[1]
                if seg_id in segments_data:
                    segments_data[seg_id]["vue"] = value

        type_list = [segments_data[key]['type'] for key in sorted(segments_data, key=int)]
        vue_list = [segments_data[key]['vue'].lower() if segments_data[key]['vue'] else None 
                   for key in sorted(segments_data, key=int)]
        
        main_enveloppe_conforme(type_list, vue_list,analysis_id)
        analysis = Analysis.query.get(analysis_id)
        return render_template("resultat.html", 
                             image_3d=analysis.enveloppe3d_img["enveloppe_3d"] ,
                             image_enveloppe=analysis.enveloppe_conforme_img["enveloppe_conforme"] )
                            #  image_3d="static/enveloppe_3d.png", 
                            #  image_enveloppe="static/enveloppe_conforme.png")
                            # ,analysis_id=analysis_id)
    
    except Exception as e:
        print(f"Erreur dans traiter_types: {str(e)}")
        #flash("Une erreur est survenue lors du traitement.", 'error')
        return render_template("erreur.html")
        # return redirect(url_for('index'))

@app.route('/download')
@login_required
def download_file():
    analysis_id = session.get('analysis_id')
    if analysis_id is None:
        return render_template("erreur.html")
    try:
        if not file_ref:
            #flash("Aucun fichier à télécharger", 'error')
            return redirect(url_for('index'))

        file_to_download = main_extrusion(analysis_id, file_ref)
        return send_file(file_to_download, as_attachment=True)
    except Exception as e:
        print(f"Erreur de téléchargement: {str(e)}")
        #flash("Erreur lors du téléchargement du fichier", 'error')
        return redirect(url_for('index'))

# Autres routes
@app.route("/viewer")
@login_required
def viewer():
    return render_template("viewer.html")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=False)