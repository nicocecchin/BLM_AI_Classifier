from flask import Flask, render_template, request, jsonify
from vec_db_functions import vector_search
import sys
import os
from insertion import get_suggested_descriptions

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from retrievers.Bm25 import Bm25
from retrievers.Fuzzy import Fuzzy
from retrievers.Sbert import Sbert
from retrievers.HybridRetriever import HybridRetriever

app = Flask(__name__)

bm25 = Bm25("../datasets/catalogue_01.csv", 10)
fuzzy = Fuzzy("../datasets/catalogue_01.csv", 10, method='token_sort_ratio')
sbert_512 = Sbert("../datasets/catalogue_01.csv", 10, size=512)
sbert_1024 = Sbert("../datasets/catalogue_01.csv", 10, size=1024)
hybrid1 = HybridRetriever("../datasets/catalogue_01.csv", 10, retriever_name="bm25", ranker_name="cross_encoder", model_name="ms-marco-MiniLM-L-6-v2")
hybrid2 = HybridRetriever("../datasets/catalogue_01.csv", 10, retriever_name="bm25", ranker_name="cross_encoder", model_name="ms-marco-TinyBERT-L-2-v2")
hybrid3 = HybridRetriever("../datasets/catalogue_01.csv", 10, retriever_name="sbert_1024", ranker_name="cross_encoder", model_name="ms-marco-MiniLM-L-6-v2")

models = ['bm25', 'fuzzy', 'sbert_512', 'sbert_1024', 'hybrid1', 'hybrid2', 'hybrid3']
model = bm25 # Default model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def list_models():
    return jsonify(models)

@app.route('/set_model', methods=['POST'])
def set_model():
    selected_model = request.json.get('model')
    if selected_model:
        if selected_model == 'bm25':
            global model
            model = bm25
        elif selected_model == 'fuzzy':
            model = fuzzy
        elif selected_model == 'sbert_512':  
            model = sbert_512
        elif selected_model == 'sbert_1024':
            model = sbert_1024
        elif selected_model == 'hybrid1':
            model = hybrid1
        elif selected_model == 'hybrid2':
            model = hybrid2
        elif selected_model == 'hybrid3':
            model = hybrid3
        else:
            return jsonify({"status": "error", "message": "Unknown model"}), 400
        # Set the model for the current session or user
        print(f"Model set to: {selected_model}")
        return jsonify({"status": "success", "model": selected_model})
    return jsonify({"status": "error", "message": "No model specified"}), 400

@app.route('/get_results', methods=['POST'])
def get_results():
    data = request.get_json()
    user_input = data.get('input', '')
    
    results = model.retrieve(user_input)

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