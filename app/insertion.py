from APIs import models, get_llm_explanation
import time
import tiktoken

model = "Meta-Llama-3.1-405B-Instruct"

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_suggested_descriptions(user_input, materials):
    message = "You will be presented 10 examples of Italian and English descriptions no longer than 40 characters. These descriptions represent material used by a company."
    formatted_materials = [
        {
            'description_ita': m["it"],
            'description_eng': m["en"],
        }
        for m in materials
    ]
    message += '\n'.join([str(item) for item in formatted_materials])
    
    message += "\nThe following is a description input by a user, it might be longer than 40 characters and not coherent with the style of the previous materials:\n"
    message += user_input
    message += "\nYour goal is to provide exactly 5 Italian descriptions and 5 English descriptions, each within 40 characters, based on the user's input and matching the style of the respective examples. Output format:\n- Only 10 lines total.\n- The first 5 lines must be the Italian descriptions.\n- The next 5 lines must be the English descriptions.\n- Do not include any explanations or extra text.\n- Each description must be on its own line, plain text only."
    
    print(message)
    start_time = time.time()
    token_count = count_tokens(message, model="gpt-4o")
    print(f"Tokens in the prompt: {token_count}")
    print(f"Tokens counted in: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    response = (get_llm_explanation(model, message))
    print(f"Descriptions for model {model} generated in: {time.time() - start_time:.2f} seconds")
   
    print(f"\n--- Output for model: {model} ---")
    if "Error" in response or "error" in response.lower():
        print(f"Error: {response}")
    
    # Filter out blank lines
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    
    if len(lines) != 10:
        print(f"Warning: Expected 10 lines, got {len(lines)}")
        print(response)
    else:
        print("\nItalian descriptions:")
        for line in lines[:5]:
            print(f"- {line}")
        print("\nEnglish descriptions:")
        for line in lines[5:10]:
            print(f"- {line}")
    
    # Return the output, handling cases where we might have fewer than 10 lines
    ita = lines[:5] if len(lines) >= 5 else lines
    eng = lines[5:10] if len(lines) >= 10 else lines[5:] if len(lines) > 5 else []
    
    return (ita, eng)