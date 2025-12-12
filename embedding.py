from transformers import DebertaTokenizer, DebertaConfig, DebertaModel
import csv
import numpy as np
import torch
import argparse

# Location of the pre-downloaded tokenizer and model 
cachepath = "/courses/nchamber/nlp/huggingface"
model_name = 'microsoft/deberta-base-mnli'

# Now load the tokenizer
print('Loading the tokenizer...')
tokenizer = DebertaTokenizer.from_pretrained(model_name, cache_dir=cachepath, local_files_only=True)

print('Initializing the model...')
config = DebertaConfig.from_pretrained(model_name, output_hidden_states=True, cache_dir=cachepath, local_files_only=True)
model = DebertaModel.from_pretrained(model_name, config=config, cache_dir=cachepath, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:",device)
model = model.to(device)

def embed_sentence(str):
    # returns a tensor embedding of the string.
    inputs = tokenizer([str], padding=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0] # [0] grabs the first sentence in the given list

    return last_hidden_states[0]


def save_tensor_embeddings(file_path):
    """
    Reads a TSV file and embeds each speech.

    Args:
        file_path (str): The path to the TSV file.

    Returns:
        A list of tensor embeddings of speeches.
    """
    nump_list = []
    # ten_list = []
    rows = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as tsvfile:
            # Create a csv.reader object with tab as the delimiter
            lines = tsvfile.readlines()

            # Iterate through each row in the TSV file
            for line in lines:
                line = line.split('\t')
                # print(f'Before: Lenght: {len(line[2])}, Text: {line[2]}')
                if len(line[2]) > 10000:
                    text = line[2][0:10000]
                else:
                    text = line[2]
                # print(f'After: Lenght: {len(text)}, Text: {text}\n')

                speech_tensor = embed_sentence(text)
                speech_tensor_cpu = speech_tensor.detach().cpu()
                del speech_tensor # To keep memory free on the gpu (from ChatGPT)
                # ten_list.append(speech_tensor_cpu)
                nump_list.append(speech_tensor_cpu.numpy()) # .cpu suggested in error message
                rows += 1
                if rows % 1000 == 0:
                    print(f'{rows} rows have been processed...')

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f'There were {rows} rows processed.')

    return np.array(nump_list)


if __name__ == "__main__":
    tsv_paths = ["sample_executive_orders.tsv","sample_spoken_records.tsv"]
    matrix_filenames = ['executive_orders_matrix.npy',"spoken_records_matrix.npy"]

    # Argument parser
    parser = argparse.ArgumentParser(description="LMTester.py")
    parser.add_argument('-model', type=str, choices=['exorders','speeches'], default='exorders', help='Type of document to search')

    # Parse the command-line arguments.
    args = parser.parse_args()
    print(args)

    # Set variables:
    if args.model == 'exorders':
        i = 0
    elif args.model == 'speeches':
        i = 1

    print('Saving tensors...')
    nump_matrix = save_tensor_embeddings(tsv_paths[i])
    print(f'There are {len(nump_matrix)} tensor embeddings stored.')

    # torch.save(tensor_list, 'executive_orders_tensors.pt') # Taken from google gen AI
    np.save(matrix_filenames[i], nump_matrix) # Taken from google gen AI