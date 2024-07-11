from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import torchaudio
import re
from string import Template
prompt_template = Template("Human: ${inst} </s> Assistant: ")
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained("m-a-p/ChatMusician", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("m-a-p/ChatMusician", torch_dtype=torch.float16, device_map="cpu", resume_download=True).eval()
# Define the API endpoint and JSON payload
$uri = "http://localhost:5000/ask-ollama"
$jsonBody = @{
    prompt = "Create ABC notation for a simple melody"
} | ConvertTo-Json

# Make the POST request
$response = Invoke-RestMethod -Uri $uri -Method Post -Body $jsonBody -ContentType "application/json"

# Display the response
$response

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    min_new_tokens=10,
    max_new_tokens=1536
)

instruction = """Develop a musical piece using the given chord progression.
'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'
"""

prompt = prompt_template.safe_substitute({"inst": instruction})
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
response = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs['attention_mask'].to(model.device),
        eos_token_id=tokenizer.eos_token_id,
        generation_config=generation_config,
        )
response = tokenizer.decode(response[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)

# to render abc notation, you need to install symusic
# pip install symusic
from symusic import Score, Synthesizer, BuiltInSF3, dump_wav

abc_pattern = r'(X:\d+\n(?:[^\n]*\n)+)'
abc_notation = re.findall(abc_pattern, response+'\n')[0]
s = Score.from_abc(abc_notation)
audio = Synthesizer().render(s, stereo=True)
torchaudio.save('cm_music_piece.wav', torch.FloatTensor(audio), 44100)
