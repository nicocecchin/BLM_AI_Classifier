
import pandas as pd


def llm_results(input_folder, output_file, dataset):
    import os

    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"The input folder {input_folder} does not exist.")

    # Prepare the output file
    with open(output_file, 'w') as f:
        f.write("model,qualitative_eval_score,characters_respected,format_respected\n")

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.startswith(dataset) and filename.endswith('.csv'):
        # if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            data = pd.read_csv(file_path)
            
            model_name = filename.replace('.csv', '')
            qualitative_mean = data['qualitative_eval_score'].mean()
            characters_respected_pct = (data['characters_respected'] == True).mean()
            format_respected_pct = (data['format_respected'] == True).mean()

            with open(output_file, 'a') as f:
                f.write(f"{model_name},{qualitative_mean},{characters_respected_pct},{format_respected_pct}\n")





    print(f"Evaluation results saved to {output_file}")

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM results.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing queries and retrieved items.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file to save evaluation results.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset being evaluated.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    llm_results(args.input_folder, args.output_file, args.dataset)
