from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  

llm = Ollama(model="llama3")



@app.route('/ask-ollama', methods=['POST'])
def ask_ollama():

    data = request.get_json()
    prompt = data.get('prompt', '')
    
    


    response = llm.invoke(prompt)
    print(response)
 
    try:


        if response:
            return jsonify( response)
        else:
            return jsonify({'error': 'No ABC notation found in the response'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to parse response: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
