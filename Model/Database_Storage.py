# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:11:56 2024

@author: 7muha
"""
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2.extras import RealDictCursor 

SQLALCHEMY_DATABASE_URL='postgresql://postgres:123@localhost/BankClassifier'
engine = create_engine(SQLALCHEMY_DATABASE_URL)

Session = sessionmaker(autocommit=False,autoflush=False, bind=engine)

Base = declarative_base()
'''
try:
    conn = psycopg2.connect(host='localhost', database="BankClassifier", user="postgres", password='123', cursor_factory='RealDictCursor')
    print('database connnected')   
    cursor=conn.cursor()
except:
    print('database cannot be connected')


cursor.execute("""CREATE  """)
'''
