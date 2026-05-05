from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from llm_agent_py import SQLAgent

app = Flask(__name__)
app.static_folder = 'static'
CORS(app)

sql_agent = SQLAgent()
sql_agent.ml_agent.pretrain_all_models()

@app.route('/')
def index():
    return render_template('templates_folder.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        sql_query, response, error = sql_agent.process_question(question)

        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'question': question,
            'sql_query': sql_query,
            'response': response,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    try:
        conn = sql_agent.db_manager.get_connection()
        if conn:
            conn.close()
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)