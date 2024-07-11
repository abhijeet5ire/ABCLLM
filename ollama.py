from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)


llm = Ollama(model="llama3")


def is_abc_prompt(prompt):
    abc_keywords = ['ABC notation', 'music notation', 'sheet music', 'musical score']
    for keyword in abc_keywords:
        if keyword.lower() in prompt.lower():
            return True
    return False

@app.route('/ask-ollama', methods=['POST'])
def ask_ollama():
    # Get prompt from request JSON
    data = request.get_json()
    prompt = data.get('prompt', '')


    if not is_abc_prompt(prompt):
        return jsonify({'error': 'Prompt must be related to ABC notation'}), 400


    response = llm.invoke(prompt+'only return abc notation no other text is required')
    print(response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
