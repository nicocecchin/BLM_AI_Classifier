from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import pandas as pd
import os
import sys
import argparse
import datetime
# retriever_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../retrievers'))
# sys.path.append(retriever_path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from retrievers.Retriever import Retriever
from retrievers.Bm25 import Bm25
from retrievers.Sbert import Sbert
from retrievers.Qwen import Qwen
from retrievers.Random import Random
from retrievers.Fuzzy import Fuzzy

def get_model(model_name: str, catalogue: str, output_length: int) -> Retriever:
    if model_name == 'bm25':
        return Bm25(data_source=catalogue, output_length=output_length)
    elif model_name == 'sbert_512':
        return Sbert(data_source=catalogue, output_length=output_length, size=512)
    elif model_name == 'sbert_768':
        return Sbert(data_source=catalogue, output_length=output_length, size=768)
    elif model_name == 'sbert_1024':
        return Sbert(data_source=catalogue, output_length=output_length, size=1024)
    elif model_name == 'qwen_1024':
        return Qwen(data_source=catalogue, output_length=output_length, size=1024)
    elif model_name == 'qwen_2560':
        return Qwen(data_source=catalogue, output_length=output_length, size=2560)
    elif model_name == 'qwen_4096':
        return Qwen(data_source=catalogue, output_length=output_length, size=4096)
    elif model_name == 'random':
        return Random(data_source=catalogue, output_length=output_length, random_seed=123)
    elif model_name == 'fuzzy_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='ratio')
    elif model_name == 'fuzzy_sort_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_sort_ratio')
    elif model_name == 'fuzzy_set_ratio':
        return Fuzzy(data_source=catalogue, output_length=output_length, method='token_set_ratio')
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

        ids = [item.item_id for item, _ in results[0:k]]
        if positive_id in ids:
            rank = ids.index(positive_id) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
        
    map_k = sum(scores) / len(scores)
    avg_time = sum(times) / len(times)
    return {"score": map_k, 'avg_time': avg_time}

def evaluate_model_on_datasets(dataset_paths: List[str], dataset_name: str, catalogue_paths: List[str], model_name: str, k: int) -> List[Tuple[str, float, float]]:
    results = []
    for dataset_path, catalogue_path in zip(dataset_paths, catalogue_paths):
        dataset = pd.read_csv(dataset_path, sep=',')
        dataset = dataset.head(100) # limit to 100 rows for testing
        model = get_model(model_name, catalogue_path, k)
        score_k_data = compute_map_k(model, dataset, k)
        results.append((dataset_name, score_k_data['score'], score_k_data['avg_time']))
    return results

def read_arguments():
    parser = argparse.ArgumentParser(description="Python to evaluate a retriever with respect to a dataset and a catalogue.")
    
    parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--catalogue', type=str, required=True, help='Path of the catalogue.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--model', type=str, required=True, choices=['bm25', 'sbert_512', 'sbert_768', 'sbert_1024', 'qwen_1024', 'qwen_2560', 'qwen_4096', 'random', 'fuzzy_ratio', 'fuzzy_sort_ratio', 'fuzzy_set_ratio'], help='Algorithm model to use for retrieval [bm25, sbert_512, sbert_1024, random, fuzzy_ratio, fuzzy_sort_ratio, fuzzy_set_ratio].')
    parser.add_argument('--k', type=int, help='MAP@k metric, number of results to consider for the Medium Average Precision (MAP) calculation.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    dataset_path = args['dataset']
    dataset_name = args['dataset_name']
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{args['output_folder']}/{current_datetime}"
    catalogue_path = args['catalogue']
    model = args['model']
    k = args['k']

    os.makedirs(output_folder, exist_ok=True)
    results = evaluate_model_on_datasets(
        dataset_paths=[dataset_path],
        dataset_name=dataset_name,
        catalogue_paths=[catalogue_path],
        model_name=model,
        k=k
    )

    # print the results
    for dataset, mapk, avg_time in results:
        print(f"Dataset: {dataset}, MAP@{k}: {mapk:.4f}, Avg Time: {avg_time:.4f} seconds")

    # save the results to a CSV file
    output_file = os.path.join(output_folder, f"results_{model}_k_{k}.csv")
    results_df = pd.DataFrame(results, columns=['Dataset', f'MAP@{k}', 'Avg Time (seconds)'])
    results_df.to_csv(output_file, index=False)
    print(f"Evaluation completed. Output saved in {output_folder}.")