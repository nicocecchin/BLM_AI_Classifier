from flask import Flask, render_template, request, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from rel_db_functions import add_materials, add_long_description, reset_db, db, bm25_search
from vec_db_functions import create_vector_db, vector_search, create_collection
import time
import nltk
from insertion import get_suggested_descriptions

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://blmuser:BLM_AI_Classifier@localhost/blmdb'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)

# with app.app_context():
#     db.create_all()  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_results', methods=['POST'])
def get_results():
    data = request.get_json()
    user_input = data.get('input', '')

    # results = bm25_search(user_input)
    results = vector_search(user_input)
    # print(results)
    # get_suggested_descriptions(user_input, results)

    formatted_results = [
        {
            'id': r["material_id"],
            'description_ita': r["it"],
            'description_eng': r["en"],
            'score': r["score"]
        }
        for r in results
    ]

    # print(formatted_results)
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

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    data = request.get_json()
    user_input = data.get('input', '')
    results = vector_search(user_input)
    (ita, eng) = get_suggested_descriptions(user_input, results)

    return jsonify({'ita': ita, 'eng': eng})

if __name__ == '__main__':
    # with app.app_context():
    #     reset_db()
    #     start = time.time()
    #     add_materials()
    #     end = time.time()
    #     print(f"Materials added in {end - start:.4f} seconds")
        
    #     start = time.time()
    #     add_long_description()
    #     end = time.time()
    #     print(f"Long descriptions added in {end - start:.4f} seconds")

    # with app.app_context():
    #     start = time.time()
    #     create_collection()
    #     create_vector_db()
    #     end = time.time()
    #     print(print(f"Time taken to create vector database: {end - start} seconds") )

    
    app.run()