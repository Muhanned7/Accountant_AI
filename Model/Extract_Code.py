# -*- coding: utf-8 -*-
"""
Created on Sat May 25 07:41:24 2024

@author: usa
"""
import PyPDF2
import re
import pdfplumber
import pandas as pd
import csv
import os
import glob


"""create classifier to parse bank statements
and classify them into outside food, groceries, miscelleneous, Rent, family, 
and loans."""

 
def refine_data(descriptions,data):
       pattern_one = r"([^\n]+(?:\n[^\n]+)?)\n(\d+/\d+)\s+([\d.]+)\s+([^\n]+(?:\n[^\n]+)?)"

       for desc in data:
           c = re.findall(pattern_one, desc[2])
           if len(c)>=1:
            pattern = r"^(0[1-9]|1[0-2])/(0[1-9]|[1-2][0-9]|3[0-1]).*"
            matches = re.findall(pattern, desc[2])
            print(matches)
            for match in matches:
                
                desc[2] = re.sub(pattern,'', desc[2])
                date, amount, description = match 
                data.append([date.strip(), amount.strip(), description.strip()])
#                print(data)
 

def read_pdf(file_path):
    with open(file_path,'rb') as file:
        
        reader = PyPDF2.PdfReader(file)
        text=''
        for page in reader.pages:
            text +=page.extract_text()
            
        return text


if __name__ == "__main__":
    # Specify the directory containing the PDF files
    directory = "D:\ Bank_Statement\Training_Data"  # Replace 'pdf_files' with your directory name

    # Use glob to find all PDF files in the directory
    data = []   
    descriptions =[]
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    for pdf_file in pdf_files:
        file_path = pdf_file  # Provide the path to your PDF file
        text = read_pdf(file_path)

        
        #extract tag and amount
        '''pattern =   r"(\d{1,2}/\d{1,2})\s+([\d.]+)\s+((?:[^\n]*(?!\n\d+/\d+\s)[^\n]*\n?){1,2})"
        
        matches = re.findall(pattern, text) 
        for match in matches:
            date, amount, description = match
            descriptions.append(description)  
            data.append([date.strip(), amount.strip(), description.strip()])
          #extract tag and amount
          #pattern =   r"(\d{1,2}/\d{1,2})\s+([\d.]+)\s+((?:[^\n]*(?!\n\d+/\d+\s)[^\n]*\n?){1,2})"
        '''
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
            # Specify the CSV file path
        csv_file_path = "extracted_data_training_new.csv"
    
        
   

    
  
    # Write the data to a CSV file
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Date", "Amount", "Description"])  # Write the header row
            writer.writerows(data)                
                        