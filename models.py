# from flask_sqlalchemy import SQLAlchemy

# db = SQLAlchemy()  # Note: db est maintenant ici


# # Modèle User
# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(100), unique=True, nullable=False)
#     password_hash = db.Column(db.String(200))
    
#     def set_password(self, password):
#         self.password_hash = generate_password_hash(password)
    
#     def check_password(self, password):
#         return check_password_hash(self.password_hash, password)
    



# #Modèle Analyse
# class Analysis(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     status = db.Column(db.String(20), default='uploaded')  # uploaded/processing/completed/failed
#     original_filename = db.Column(db.String(200))
#     building_height = db.Column(db.Float)
#     parcelle_data = db.Column(db.JSON)  # Stockera {points: [{x,y,z}, ...]}
#     segments_data = db.Column(db.JSON)   # Stockera {segments: [{x1,y1,x2,y2,height}, ...]}
    
#     # Relation
#     user = db.relationship('User', backref='analyses')
#     segments = db.relationship('Segment', backref='analysis', cascade='all, delete-orphan')

# class Segment(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=False)
#     segment_id = db.Column(db.Integer)  # Numéro du segment (0,1,2...)
#     x1 = db.Column(db.Float)
#     y1 = db.Column(db.Float)
#     x2 = db.Column(db.Float)
#     y2 = db.Column(db.Float)
#     height = db.Column(db.Float)
#     type = db.Column(db.String(20), nullable=True)  # Ex: "VO"
#     offset = db.Column(db.Float, nullable=True)     # Ex: 2.0

# class RawCoordinates(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=False)
#     latitude = db.Column(db.Float)
#     longitude = db.Column(db.Float)
#     altitude = db.Column(db.Float, nullable=True)
#     is_original = db.Column(db.Boolean, default=True)


from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)  # Index pour accès rapide
    password_hash = db.Column(db.String(200))
    # created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relations
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    # status = db.Column(db.String(20), default='uploaded', nullable=False)
    # original_filename = db.Column(db.String(200))
    # building_height = db.Column(db.Float)
    parcelle_data = db.Column(db.JSON)
    parcelle_coords = db.Column(db.JSON)
    # segments_data = db.Column(db.JSON)
    enveloppe_finale = db.Column(db.JSON)
    enveloppe3d_img = db.Column(db.JSON) 
    enveloppe_conforme_img = db.Column(db.JSON) 
    ifc_file = db.Column(db.JSON) 
    # created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    # updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    
    # Relations
    # segments = db.relationship('Segment', backref='analysis', lazy='dynamic', cascade='all, delete-orphan')
    # raw_coordinates = db.relationship('RawCoordinates', backref='analysis', cascade='all, delete-orphan')

# class Segment(db.Model):
#     __tablename__ = 'segments'
    
#     id = db.Column(db.Integer, primary_key=True)
#     analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False)
#     segment_id = db.Column(db.Integer, index=True)  # Index pour tri/accès fréquent
#     x1 = db.Column(db.Float, nullable=False)
#     y1 = db.Column(db.Float, nullable=False)
#     x2 = db.Column(db.Float, nullable=False)
#     y2 = db.Column(db.Float, nullable=False)
#     height = db.Column(db.Float, nullable=False)
#     type = db.Column(db.String(20))
#     offset = db.Column(db.Float)
    
#     __table_args__ = (
#         db.Index('idx_segment_analysis', 'analysis_id', 'segment_id'),  # Index composite
#     )

# class RawCoordinates(db.Model):
#     __tablename__ = 'raw_coordinates'
    
#     id = db.Column(db.Integer, primary_key=True)
#     analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False)
#     latitude = db.Column(db.Float, nullable=False)
#     longitude = db.Column(db.Float, nullable=False)
#     altitude = db.Column(db.Float)
#     is_original = db.Column(db.Boolean, default=True, nullable=False)
    
#     __table_args__ = (
#         db.Index('idx_coords_analysis', 'analysis_id', 'is_original'),  # Index optimisé
#     )
