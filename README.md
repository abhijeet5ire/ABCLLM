# LLM to Generate ABC Notation Music

This project contains a Flask server that connects with the Ollama -> Llama 3 model (other models can also be used).

## How to Install

### 1. Install Ollama

Run the following command to install Ollama:
```sh
curl -fsSL https://ollama.com/install.sh | sh
```
### 2. Install the Llama 3 Model
```sh
ollama run llama3
```

### 3. Start the Ollama Server
```sh
ollama serve

```

### 4. Install Python Dependencies
```py
pip install -r requirements.txt
```

### 5. Run the Flask Server
```py
python ollama.py
```