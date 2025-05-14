from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_results', methods=['POST'])
def get_results():
    data = request.get_json()
    user_input = data.get('input', '')
    results = [f"{user_input} risultato {i+1}" for i in range(10)]
    return jsonify(results)


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
    app.run(debug=True)