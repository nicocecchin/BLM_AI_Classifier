from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from retrievers.Item import Item
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import bm25s
import time
import csv

class Bm25(Retriever):
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
                # reader = csv.reader(file, delimiter=';')
                reader = csv.reader(file, delimiter=',')
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

            # tokenize the documents and calculate BM25 scores
            corpus = [doc[1].lower() for doc in docs]
            retriever = bm25s.BM25(corpus=corpus)
            retriever.index(bm25s.tokenize(corpus), show_progress=False)
            results, scores = retriever.retrieve(bm25s.tokenize(query.lower()), k=self.output_length*2, show_progress=False)

            # filter the results to return only unique materials
            # and limit the number of results to output_length
            output = []
            seen = set()
            for i, r in enumerate(results[0]):
                for d in docs:
                    if len(output) >= self.output_length:
                        break
                    if d[1].lower() == r  and d[0] not in seen:
                        item = Item(
                            item_id=d[0],
                            ita_short_desc=self.Material.query.filter_by(id=d[0]).first().short_desc_it,
                            eng_short_desc=self.Material.query.filter_by(id=d[0]).first().short_desc_eng,
                        )
                        output.append((item, scores[0][i]))
                        seen.add(d[0])
                        break
            
            end = time.time()

            return output, end-start