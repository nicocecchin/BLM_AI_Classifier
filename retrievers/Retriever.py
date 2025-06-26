from typing import Dict, List, Set, Tuple
from Item import Item
import os
import argparse

class Retriever:
    def __init__(self, data_source: str, output_length: int):
        self.data_source = data_source
        self.output_length = output_length

    def retrieve(self, query:str) -> Tuple[List[Tuple[Item, float]], float]:
        pass
    
def read_arguments():
    parser = argparse.ArgumentParser(description="Python class that retrieves data based on a user query.")
    
    parser.add_argument('--catalogue', type=str, required=True, help='Path to the data source file.')
    parser.add_argument('--query', type=str, required=True, help='User query.')
    parser.add_argument('--model', type=str, required=True, help='Algorithm model to use for retrieval [bm25, sbert_512, sbert_1024].')
    parser.add_argument('--output_length', type=int, required=True, help='Number of results to return.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    # retrievers
    from Bm25 import Bm25

    args = read_arguments()
    catalogue = args['catalogue']
    query = args['query']
    model = args['model']
    output_length = args['output_length']

    if model == 'bm25':
        retriever = Bm25(catalogue, output_length)
    
    results, time_taken = retriever.retrieve(query)
    for i,r in enumerate(results):
        print(f"[{i+1}] item: {r[0]}, score: {r[1]}")
    print(f"Time taken: {time_taken:.4f} seconds")