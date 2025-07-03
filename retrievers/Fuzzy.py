from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from retrievers.Item import Item
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from thefuzz import fuzz
from thefuzz import process
import time
import csv

class Fuzzy(Retriever):
    def __init__(self, data_source: str, output_length: int):
        super().__init__(data_source, output_length)
        
        # create and configure the Flask app
        self.app:Flask.app.Flask = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        # create database sql from the data source
        self.db:SQLAlchemy.extension.SQLAlchemy = SQLAlchemy(self.app)

        # material table inside the app context
        with self.app.app_context():
            class Material(self.db.Model):
                id = self.db.Column(self.db.String, primary_key=True)
                short_desc_it = self.db.Column(self.db.String, primary_key=False)
                short_desc_eng = self.db.Column(self.db.String, primary_key=False)
                long_desc_it = self.db.Column(self.db.String, primary_key=False)
                long_desc_eng = self.db.Column(self.db.String, primary_key=False)
            self.Material = Material

            # (re)create the tables
            self.db.drop_all()
            self.db.create_all()

            # populate SQL database with items from the catalogue
            with open (self.data_source) as file:
                reader = csv.reader(file, delimiter=';')
                next(reader)
                for row in reader:
                    material = self.Material(
                        id=row[0],
                        short_desc_it=row[1],
                        short_desc_eng=row[2],
                        long_desc_it=row[3],
                        long_desc_eng=row[4]
                    )
                    self.db.session.add(material)
                self.db.session.commit()

    def retrieve(self, query:str) -> Tuple[List[Tuple[Item, float]], float]:
        with self.app.app_context():
            start = time.time()

            # retrieve all materials from the database
            docs = []
            for material in self.Material.query.all():
                ita = material.short_desc_it.strip()
                eng = material.short_desc_eng.strip()

                if material.long_desc_it:
                    ita = ita + ' ' + material.long_desc_it.strip()
                if material.long_desc_eng:
                    eng = eng + ' ' + material.long_desc_eng.strip()
                docs.append((material.id, ita, ''))
                docs.append((material.id, eng, ''))

            corpus = [doc[1].lower() for doc in docs]
            retriever = process.extract(query.lower(), corpus, scorer=fuzz.token_sort_ratio, limit=self.output_length*2)

            output = []
            seen = set()
            for item in retriever:
                r = item[0]
                score = item[1] 
                if len(output) >= self.output_length:
                    break
                for d in docs:
                    if (d[1].lower() == r or d[2].lower() == r) and d[0] not in seen:
                        item_obj = Item(
                            item_id=d[0],
                            ita_short_desc=self.Material.query.filter_by(id=d[0]).first().short_desc_it,
                            eng_short_desc=self.Material.query.filter_by(id=d[0]).first().short_desc_eng,
                        )
                        output.append((item_obj, score))
                        seen.add(d[0])
                        break
            
            end = time.time()

            return output, end-start