from flask import Flask, render_template, request, jsonify
from vec_db_functions import vector_search
import sys
import os
from insertion import get_suggested_descriptions


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from retrievers.Bm25 import Bm25
from retrievers.OpenSearch import OpenSearch

app = Flask(__name__)

bm25 = Bm25("../datasets/catalogue_01.csv", 10)
opensearch = OpenSearch("../datasets/catalogue_01.csv", 10)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_results', methods=['POST'])
def get_results():
    data = request.get_json()
    user_input = data.get('input', '')

    results = opensearch.retrieve(user_input)
    print(f"Time taken to retrieve results: {results[1]}")

    formatted_results = [
        {
            'id': item[0].item_id,
            'description_ita': item[0].ita_short_desc,
            'description_eng': item[0].eng_short_desc,
            'score': "{:.4f}".format(item[1]),
        }
        for item in results[0]
    ]

    print(formatted_results)
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
    app.run()