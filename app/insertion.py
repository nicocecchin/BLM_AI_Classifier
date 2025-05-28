from APIs import models, get_llm_explanation
import time
import tiktoken

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


    # ita = []
    # eng = []
    # for m in materials:
    #     ita.append(m["it"])
    #     eng.append(m["en"])
    
    # message += '\nItalian descriptions:\n'
    # message += '\n'.join([str(item) for item in ita])
    # message += '\nEnglish descriptions:\n'
    # message += '\n'.join([str(item) for item in eng])

    message += "\nThe following is a description input by a user, it might be longer than 40 characters and not coherent with the style of the previous materials:\n"

    message += user_input

    message += "\nYour goal is to provide exactly 5 Italian descriptions and 5 English descriptions, each within 40 characters, based on the user's input and matching the style of the respective examples. Output format:\n- Only 10 lines total.\n- The first 5 lines must be the Italian descriptions.\n- The next 5 lines must be the English descriptions.\n- Do not include any explanations or extra text.\n- Each description must be on its own line, plain text only." 

    print(message)

    start_time = time.time()
    token_count = count_tokens(message, model="gpt-4o")
    print(f"Tokens in the prompt: {token_count}")
    print(f"Tokens counted in: {time.time() - start_time:.2f} seconds")

    response = []
    for model in models:
        start_time = time.time()
        response.append(get_llm_explanation(model, message))
        print(f"Descriptions for model {model} generated in: {time.time() - start_time:.2f} seconds")
    
    for model_name, output in zip(models, response):
        print(f"\n--- Output for model: {model_name} ---")
        if "Error" in output or "error" in output.lower():
            print(f"❌ Error: {output}")
            continue

        lines = output.strip().splitlines()
        if len(lines) < 10:
            print(f"⚠️ Warning: Expected 10 lines, got {len(lines)}")
            print(output)
        else:
            print("\nItalian descriptions:")
            for line in lines[:5]:
                print(f"- {line}")
            print("\nEnglish descriptions:")
            for line in lines[5:10]:
                print(f"- {line}")

    # Ritorna SEMPRE l’output del primo modello
    first_valid = response[0].strip().splitlines()

    ita = first_valid[:5]
    eng = first_valid[5:10]

    return (ita, eng)

    