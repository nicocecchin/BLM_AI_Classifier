import argparse
import time
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from retrievers.Item import Item
from retrievers.Sbert import Sbert

from app.insertion import get_suggested_descriptions, _load_config
from dotenv import load_dotenv
from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import RetryPolicy
from ollama import chat
import json

def get_qualitative_evaluation(query: str, retrieved_items: List[Tuple[Item, float]], suggested_ita: List[str], suggested_eng: List[str]):
    start = time.time()
    evaluator_prompt = "You will be shown 10 examples of Italian item descriptions and their corresponding English translations. Each description is 40 characters or fewer and represents an item from a company's catalogue.\n"
    evaluator_prompt += "Your task:\n"
    evaluator_prompt += "Given a user's input (which may be longer than 40 characters and may not follow the same format) which can be in Italian or in English, generate:\n"
    evaluator_prompt += "- 5 Italian descriptions and\n"
    evaluator_prompt += "- 5 English descriptions\n"
    evaluator_prompt += "Each must be:\n"
    evaluator_prompt += "- No longer than 40 characters\n"
    evaluator_prompt += "- Matched in format and style to the examples provided.\n"
    evaluator_prompt += "- Derived from the user's input\n"
    evaluator_prompt += "Output format:\n"
    evaluator_prompt += "- Exactly 10 lines total\n"
    evaluator_prompt += "- First 5 lines: Italian descriptions\n"
    evaluator_prompt += "- Next 5 lines: English translations\n"
    evaluator_prompt += "- One description per line\n"
    evaluator_prompt += "- No additional text, headers, or explanation\n"
    evaluator_prompt += "Note: These descriptions that you must generate are proposals for a new item to be added to the catalogue.\n\n"

    # Provided examples
    evaluator_prompt += "Examples:\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x16-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x16-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x20-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x20-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M4x16-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M4x16-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M5x16-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M5x16-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x30-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x30-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x10-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x10-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M6x25-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M6x25-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M5x60-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M5x60-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M4x20-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M4x20-70'}\n"
    evaluator_prompt += "{'description_ita': 'VITE SICUREZZA TCBEI+PIN A2 M5x20-70', 'description_eng': 'SAFETY SCREW HSBH+PIN A2 M5x20-70'}\n\n"

    # Example user input and expected demonstration
    evaluator_prompt += "Example user input:\n"
    evaluator_prompt += "vite sicurezza TCBEI A2 M6x16 80\n\n"
    evaluator_prompt += "Expected output:\n"
    evaluator_prompt += "VITE SICUREZZA TCBEI+PIN A2 M6x16-80\n"
    evaluator_prompt += "VITE SICUREZZA TCBEI+PIN A2 M6x20-80\n"
    evaluator_prompt += "VITE SICUREZZA TCBEI+PIN A2 M4x16-80\n"
    evaluator_prompt += "VITE SICUREZZA TCBEI+PIN A2 M5x16-80\n"
    evaluator_prompt += "VITE SICUREZZA TCBEI+PIN A2 M6x30-80\n"
    evaluator_prompt += "SAFETY SCREW HSBH+PIN A2 M6x16-80\n"
    evaluator_prompt += "SAFETY SCREW HSBH+PIN A2 M6x20-80\n"
    evaluator_prompt += "SAFETY SCREW HSBH+PIN A2 M4x16-80\n"
    evaluator_prompt += "SAFETY SCREW HSBH+PIN A2 M5x16-80\n"
    evaluator_prompt += "SAFETY SCREW HSBH+PIN A2 M6x30-80\n\n"

    # Provided Examples from retrieval
    formatted_materials = [
        {
            'description_ita': item.ita_short_desc,
            'description_eng': item.eng_short_desc,
        }
        for item, _ in retrieved_items
    ]
    evaluator_prompt += "Provided Examples:\n"
    evaluator_prompt += '\n'.join([str(item) for item in formatted_materials]) + "\n\n"

    # Actual user input and candidate output
    evaluator_prompt += "User input:\n" + query + "\n\n"
    evaluator_prompt += "Candidate Output:\n"
    for desc in suggested_ita:
        evaluator_prompt += desc + "\n"
    for desc in suggested_eng:
        evaluator_prompt += desc + "\n"
    evaluator_prompt += "\n"

    # Evaluation instructions
    evaluator_prompt += "You are an automated evaluation system. Your task is to judge how accurately the Candidate’s generated descriptions reflect both the user’s input and the formatting/content patterns from the provided examples.\n"
    evaluator_prompt += "Evaluate only the accuracy and consistency of the descriptions (ignore character limits and line counts).\n\n"
    evaluator_prompt += "Assign a single score from 1 to 5, where:\n"
    evaluator_prompt += "1 = Completely inaccurate/inconsistent\n"
    evaluator_prompt += "2 = Mostly inaccurate/inconsistent\n"
    evaluator_prompt += "3 = Partially accurate/consistent\n"
    evaluator_prompt += "4 = Mostly accurate/consistent\n"
    evaluator_prompt += "5 = Perfectly accurate/consistent\n\n"
    evaluator_prompt += "Your response must be a JSON object in the following format:\n"
    evaluator_prompt += "```\n"
    evaluator_prompt += "{\n"
    evaluator_prompt += '  "score": <integer 1–5>,\n'
    evaluator_prompt += '  "comment": "<brief explanation of your assessment (max 2 sentences)>"\n'
    evaluator_prompt += "}\n"
    evaluator_prompt += "```\n\n"
    evaluator_prompt += "Now evaluate the Candidate’s output based solely on description accuracy and consistency with the examples."

    params = _load_config()

    retry = RetryPolicy(retry_total=3, timeout=10)
    client = ChatCompletionsClient(
        endpoint=params["endpoint"],
        credential=AzureKeyCredential(params["token"]),
        retry_policy=retry
    )
    response:str = None
    while response is None:
        response = client.complete(
            model="Mistral-Large-2411",
            # model="Meta-Llama-3.1-405B-Instruct",
            # model="DeepSeek-V3-0324",
            # model="grok-3",
            messages=[UserMessage(content=evaluator_prompt)]
        ).choices[0].message.content

    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end].replace('$', '\n')
        result = json.loads(json_str)
        score = str(result.get("score", ""))
        comment = result.get("comment", "")
    except Exception as e:
        score = ""
        comment = f"Parsing error: {e}"

    score = score.strip()
    comment = comment.strip()
    comment = comment.replace('"', "'")  # Replace double quotes with single quotes for consistency
    comment = comment.replace('\n', ' ')  # Remove newlines for a single line comment
    comment = f'"{comment}"'

    end = time.time()

    return score, comment, end - start


def llm_evaluator(dataset_path: str, catalogue_path: str, output_file: str, model = None):
    retriever = Sbert(data_source=catalogue_path, output_length=10, size=1024)

    dataset = pd.read_csv(dataset_path)

    existing_df = pd.read_csv(output_file, usecols=['query'])
    existing_queries = set(existing_df['query'].dropna().astype(str))

    print(f"{len(existing_queries)} already processed.")   # debug

    to_process = dataset[~dataset['query'].astype(str).isin(existing_queries)]
    print(f"{len(to_process)} to process.\n")  # debug

    for _, row in tqdm(to_process.iterrows(), total=len(to_process), desc=f"Evaluating {dataset_path} with {model}"):
        query = row['query']
        language = row.get('language', 'eng')  # Default to 'eng' if 'language' is not present

        retrieved_items, _ = retriever.retrieve(query, language=language)
        suggested_ita, suggested_eng = get_suggested_descriptions(query, retrieved_items, model)
        print(f"got suggestions for query: {query}")

        characters_respected = all(len(desc) <= 40 for desc in suggested_ita + suggested_eng)
        format_respected = (len(suggested_eng) == 5) and (len(suggested_ita) == 5)
        qualitative_eval_score, qualitative_eval_comment, time_taken = get_qualitative_evaluation(query, retrieved_items, suggested_ita, suggested_eng)

        retrieved_items_str = "$".join(repr(item) for item in retrieved_items)
        retrieved_items_str = f'"{retrieved_items_str}"'

        generated_descriptions = suggested_ita + suggested_eng
        generated_descriptions = [desc.strip() for desc in generated_descriptions if desc.strip()]
        generated_descriptions_str = "$".join(generated_descriptions)
        generated_descriptions_str = f'"{generated_descriptions_str}"'

        print(f"Writing results for query: {query}")
        print(f"Retrieved items: {retrieved_items}")
        print(f"Generated descriptions: {generated_descriptions}")
        print(f"Qualitative evaluation score: {qualitative_eval_score}, comment: {qualitative_eval_comment}")
        print(f"time taken: {time_taken:.2f} seconds")

        # Ensure CSV safety: wrap any field containing a comma in double quotes
        def csv_safe(s):
            if isinstance(s, str) and ',' in s and not (s.startswith('"') and s.endswith('"')):
                return f'"{s}"'
            return s

        query = csv_safe(query)
        retrieved_items_str = csv_safe(retrieved_items_str)
        generated_descriptions_str = csv_safe(generated_descriptions_str)
        qualitative_eval_score = csv_safe(qualitative_eval_score)
        qualitative_eval_comment = csv_safe(qualitative_eval_comment)
        time_taken = csv_safe(f"{time_taken:.2f} seconds")

        with open(output_file, "a") as f:
            f.write(f"{query},{retrieved_items_str},{generated_descriptions_str},{qualitative_eval_score},{qualitative_eval_comment},{characters_respected},{format_respected},{time_taken}\n")


def read_arguments():
    parser = argparse.ArgumentParser(description="Python to evaluate a retriever with respect to a dataset and a catalogue.")
    
    parser.add_argument('--llm_dataset', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--catalogue', type=str, required=True, help='Path of the catalogue.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    
    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == "__main__":
    
    args = read_arguments()
    dataset_path = args['llm_dataset']
    catalogue_path = args['catalogue']
    output_folder = args['output_folder']

    dataset_name = "d" + dataset_path.split('/')[-1].split('.')[0].split('_')[-1]

    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini", 
        "Meta-Llama-3.1-405B-Instruct", 
        "DeepSeek-V3-0324",
        "grok-3",
    ]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for model_name in models:
        output_file = os.path.join(output_folder, f"{dataset_name}_{model_name}_results.csv")
        results_df = pd.DataFrame(
            columns=['query', 'retrieved_items', 'generated_descriptions', 'qualitative_eval_score', 'qualitative_eval_comment', 'characters_respected', 'format_respected', 'time_taken']
        )
        if not os.path.exists(output_file):
            results_df.to_csv(output_file, index=False)
        
        print(f"Evaluating {dataset_name} with {model_name}...")
        try:
            llm_evaluator(dataset_path, catalogue_path, output_file, model=model_name)
        except Exception as e:
            print(f"Error during evaluation for {dataset_name} with {model_name}: {e}")
            continue
        print(f"Evaluation for {dataset_name} with {model_name} completed. Results saved to {output_file}.")