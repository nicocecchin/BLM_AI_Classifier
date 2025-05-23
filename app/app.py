from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from db_functions import add_materials, add_long_description, reset_db, db, bm25_search
import time
import nltk

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://blmuser:BLM_AI_Classifier@localhost/blmdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_results', methods=['POST'])
def get_results():
    data = request.get_json()
    user_input = data.get('input', '')

    results = bm25_search(user_input)

    formatted_results = [
        {
            'id': r[0],
            'description': r[1],
            'score': r[2]
        }
        for r in results
    ]
    return jsonify(formatted_results)


@app.route('/insertion')
def insert():
    return render_template('insertion.html')


@app.route('/submit_insertion', methods=['POST'])
def submit_insertion():
    data = request.get_json()
    code = data.get('code', '')
    desc_it = data.get('desc_it', '')
    desc_en = data.get('desc_en', '')
    
    
    return jsonify("received insertion data", code, desc_it, desc_en)


if __name__ == '__main__':
    app.run()