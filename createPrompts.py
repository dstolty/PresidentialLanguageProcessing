import pandas as pd


def create_prompt_format(row):

        # Instruction for the model to generate text
    #instruction = f"Generate an executive order in the style of {row['author']} from around the date, {row['date']}."
    #response = row['text']
    
    #    # Instruction for the model to classify text
    instruction = f"Analyze the following text and identify its author. The options are {', '.join(AUTHORS)}."
    input_text = row['text']
    response = row['author']
        
        # combine instruction and input for classification
    instruction = f"{instruction}\n\nText: \"{input_text}\""
        
    # format SFTTrainer expects
    # '<s>' start token, eos token added by tokenizer
    return {
        "text": f"<s>[INST] {instruction} [/INST] {response} AN"}

if __name__ == '__main__':
    OUTPUT_FILE = './classification_prompts.csv'
    INPUT_FILE = 'spoken_records.tsv'

    AUTHORS = list()
    with open('all_authors.txt') as au:
        for line in au:
           AUTHORS.append(line.strip('\n'))  ## used with classification prompts

    df = pd.read_csv(INPUT_FILE,sep='\t',names=['author','date','text'])
    df['formatted_prompt'] = df.apply(create_prompt_format,axis=1)
    df[['formatted_prompt']].to_csv(OUTPUT_FILE,index=False)
    



            