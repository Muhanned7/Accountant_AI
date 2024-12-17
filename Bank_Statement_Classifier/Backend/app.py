# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:15:57 2024

@author: 7muha
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import glob
import re
import json
import csv
import gradio
import PyPDF2
from worker import speech_to_text

sys.path.append('D:\\Bank_Statement')
from Extract_Code import read_pdf,refine_data
from User_Classifer import User_classifier
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'  # Path to save files relative to `backend/`
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text = True)
    print(json.loads(data))
    data = json.loads(data)
    print(data['InputValue'])
    input_text = data['InputValue']
    
    history = '\n'.join(conversation_history)
    
    
    inputs = tokenizer.encode_plus(history, input_text, return_tensors = "pt")
    
    output = model.generate(**inputs, max_length = 60)
    print(output)
    #(tokenizer.decode(output[0],skip_special_tokens='True'))
    #tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    response = tokenizer.decode(output[0],skip_special_tokens=True).strip()
    print(response)
    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    return response
    
    
    
    


@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if a file is in the request
    print(request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Check if the file is selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file
    '''try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"File save failed: {str(e)}"}), 500
    '''
    
    directory = "D:\\Bank_Statement\\Bank_Statement_Classifier\\Backend\\uploads"  # Replace 'pdf_files' with your directory name

    # Use glob to find all PDF files in the directory
    data = [] 
    data_all=[]
    descriptions =[]
    #pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    #for pdf_file in pdf_files:
    #pdf_file= file
    #file_path = pdf_file  # Provide the path to your PDF file
    reader = PyPDF2.PdfReader(file)
    text=''
    for page in reader.pages:
        text +=page.extract_text()
        
   

    
    #extract tag and amount
    #pattern =   r"(\d{1,2}/\d{1,2})\s+([\d.]+)\s+((?:[^\n]*(?!\n\d+/\d+\s)[^\n]*\n?){1,2})"
    pattern =   r"(\d{1,2}/\d{1,2})\s+(\d+\.\d{2})(.*)"
    matches = re.findall(pattern, text) 
    count=0
    for match in matches:
        count+=1
        date, amount, description = match 
        descriptions.append(description)  
        data.append([date.strip(), amount.strip(), description.strip()])
        
        #store it in a dictionary  
        refine_data(descriptions,data)  
        if len(description) !=0:
            data_all.append({'date':date.strip(),'amount': amount.strip(),'description': description.strip(),'type':0,'serial':count})
    User_classifier(data_all) 
    return jsonify(data_all)
            


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print('processing text to speech')
    audio_binary = request.data
    text = speech_to_text(audio_binary)
    
    
    response = app.response_class(reponse= json.dump({'text':text}),
                                        status=200, mimetype='application/json')
    print(response)
    print(response.data)
    return response

            

    
if __name__ == '__main__':
    app.run(debug=False)