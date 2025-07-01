from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import pandas as pd
import os
import sys
import argparse
retriever_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../retrievers'))
sys.path.append(retriever_path)
from retrievers.Retriever import Retriever
from retrievers.Bm25 import Bm25

def get_model(model_name: str, catalogue: str, output_length: int) -> Retriever:
    if model_name == 'bm25':
        return Bm25(data_source=catalogue, output_length=output_length)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def compute_map_k(model: Retriever, dataset: pd.DataFrame, k: int) -> Tuple[float, float]:
    scores = []
    times = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Computing MAP@{k}"):
        query = row['query']
        positive_id = row['positive']
        results, retrive_time = model.retrieve(query=query)
        times.append(retrive_time)

        ids = [item.item_id for item, _ in results]
        if positive_id in ids:
            rank = ids.index(positive_id) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
        
    map_k = sum(scores) / len(scores)
    avg_time = sum(times) / len(times)
    return {"score": map_k, 'avg_time': avg_time}

def evaluate_model_on_datasets(dataset_paths: List[str], catalogue_paths: List[str], model_name: str, k: int) -> List[Tuple[str, float, float]]:
    results = []
    for dataset_path, catalogue_path in zip(dataset_paths, catalogue_paths):
        dataset = pd.read_csv(dataset_path, sep=',')
        dataset = dataset.head(100) # limit to 100 rows for testing
        model = get_model(model_name, catalogue_path, k)
        score_1_data = compute_map_k(model, dataset, 1)
        score_k_data = compute_map_k(model, dataset, k)
        dataset_name = "dataset_01"
        results.append((dataset_name, score_1_data['score'], score_k_data['score'], score_k_data['avg_time']))
    return results

def generate_latex_table(results: List[Tuple[str, float, float, float]], model_name: str, output_path: str, k: int):
    table = "\\begin{tabular}{|l|c|c|c|}\n\\hline\n"
    table += f"Dataset & MAP@1 & MAP@{k} ({model_name}) & Avg Time (s) \\\\ \\hline\n"
    for dataset, map1, mapk, avg_time in results:
        table += f"{dataset} & {map1:.4f} & {mapk:.4f} & {avg_time:.4f} \\\\ \\hline\n"
    table += "\\end{tabular}\n"

    with open(os.path.join(output_path, f"{model_name}_results.tex"), 'w') as f:
        f.write(table)

def read_arguments():
    parser = argparse.ArgumentParser(description="Python to evaluate a retriever with respect to a dataset and a catalogue.")
    
    parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--catalogue', type=str, required=True, help='Path of the catalogue.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--model', type=str, required=True, help='Algorithm model to use for retrieval [bm25, sbert_512, sbert_1024].')
    parser.add_argument('--k', type=int, help='MAP@k metric, number of results to consider for the Medium Average Precision (MAP) calculation.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    dataset_path = args['dataset']
    output_folder = args['output_folder']
    catalogue_path = args['catalogue']
    model = args['model']
    k = args['k']

    os.makedirs(output_folder, exist_ok=True)
    results = evaluate_model_on_datasets(
        dataset_paths=[dataset_path],
        catalogue_paths=[catalogue_path],
        model_name=model,
        k=k
    )

    # print the results
    for dataset, map1, mapk, avg_time in results:
        print(f"Dataset: {dataset}, MAP@1: {map1:.4f}, MAP@{k}: {mapk:.4f}, Avg Time: {avg_time:.4f} seconds")

    generate_latex_table(results, model_name=model, output_path=output_folder, k=k)
    print(f"Evaluation completed. LaTeX table saved in {output_folder}.")