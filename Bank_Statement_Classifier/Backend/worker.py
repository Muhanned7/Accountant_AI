# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:02:28 2024

@author: 7muha
"""
from openai import OpenAI
import requests

openai_client = OpenAI()

def speech_to_text(audio_binary):
    base_url='...'
    api_url = base_url + "/speech-to-text/api/v1/recognize"
    
    params= {
        'model' : 'en-US/Multimedia',
        }
    
    body = audio_binary
    
    response = requests.post(api_url, params=params, data=body).json()
    
    text='null'
    
    while bool(response.get('results')):
        print('speech to text response: ', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text',text)
        return text
    
    
#def openai_process_message(user_message):
prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
user_message= "Hi how are you?"
openai_response = openai_client.chat.completions.create(
    model= "gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
        ],
    max_tokens=4000
    )
print("openai response:", openai_response)

response_text = openai_response.choices[0].message.content
#return response_text

