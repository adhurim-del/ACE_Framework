import requests
import json
import re
import openai
from time import time, sleep
from datetime import datetime
from halo import Halo
import textwrap
import yaml


###     file operations


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


###     API functions



def send_message(bus, layer, message):
    url = 'http://127.0.0.1:900/message'
    headers = {'Content-Type': 'application/json'}
    data = {'bus': bus, 'layer': layer, 'message': message}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print('Failed to send message')



def get_messages(bus, layer):
    url = f'http://127.0.0.1:900/message?bus={bus}&layer={layer}'
    response = requests.get(url)
    if response.status_code == 200:
        messages = response.json()['messages']
        return messages
    else:
        print('Failed to get messages')



def format_messages(messages):
    formatted_messages = []
    for message in messages:
        time = datetime.fromtimestamp(message['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        bus = message['bus']
        layer = message['layer']
        text = message['message']
        formatted_message = f'{time} - {bus} - Layer {layer} - {text}'
        formatted_messages.append(formatted_message)
    return '\n'.join(formatted_messages)
    
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")  # You can replace "gpt2" with another model like "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")

def chatbot(conversation, model="Qwen-1-5B-GGUF", temperature=0.7, max_tokens=200):
    try:
        spinner = Halo(text='Thinking...', spinner='dots')
        spinner.start()
        
        inputs = tokenizer(conversation, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids, 
            max_length=max_tokens, 
            temperature=temperature, 
            num_return_sequences=1
        )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        spinner.stop()
        
        return text, len(outputs[0])
    except Exception as oops:
        print(f'\n\nError communicating with model: "{oops}"')
        sleep(5)
