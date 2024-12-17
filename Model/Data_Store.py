# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:19:44 2024

@author: 7muha
"""

from Database_Storage import engine, Session, Base
from datetime import datetime
from models import Bank_Statements
import pandas as pd


def validate_day_month(day, month):
    import calendar
    # Check if day is valid for the given month
    try:
        if 1 <= month <= 12 and 1 <= day <= calendar.monthrange(2024, month)[1]:  # Use leap year
            return True
        return False
    except ValueError:
        return False

#models.Base.metadata.create_all(bind=engine)
session =Session()
Base.metadata.create_all(engine)
df = pd.read_csv("extracted_data_training.csv")


df.rename(columns={'Date':'Date', 'Amount':'Amount', 'Description':'Description','Type':'Type' }, inplace=True)
count=0

for _,row in df.iterrows():
    
    try:
        # Try parsing with zero-padded format
       date_obj = datetime.strptime(row['Date'] + "-2024", '%d-%b-%Y')
    except ValueError:
       # If it fails, try without zero-padding
       date_obj = datetime.strptime(row['Date'] + "-2024", '%b-%d-%Y')
    record = Bank_Statements(
        Date=date_obj,
        Amount=float(row['Amount'].replace(',', '')),
        Description=str(row['Description']),
        Type=int(row['Type']) 
        ) 
    
   
    session.add(record)
session.commit()

print("Data migrated successfully!")
        
    