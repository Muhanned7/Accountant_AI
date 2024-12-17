# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:05:30 2024

@author: 7muha
"""

from sqlalchemy import Column,Integer,DATE,DECIMAL,String
from Database_Storage import Base


class Bank_Statements(Base):
    __tablename__="BankStatements"
    ID = Column(Integer,primary_key=True,autoincrement=True)
    Date = Column(DATE, nullable=False)
    Amount=Column(DECIMAL, nullable=False)
    Description=Column(String, nullable=False)
    Type =Column(Integer, nullable=False)