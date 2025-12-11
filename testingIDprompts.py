import pandas as pd


def create_prompt(item):
    ## mod
    instruction = f"Analyze the following excerpt from a speech and identify its author. The options are {', '.join(AUTHORS)}."
    input_text = item['text']
        
        # combine instruction and input for classification
    prompt = f"{instruction}\n\nExcerpt: \"{input_text}\""

    return prompt 


if __name__ == '__main__':
    ## create list of authors
    AUTHORS = list()
    with open('all_authors.txt') as au:
        for line in au:
           AUTHORS.append(line.strip('\n'))

    df = pd.read_csv('spoken_records.tsv',sep='\t',names=['author','date','text'])
    df['formatted_prompt'] = df.apply(create_prompt,axis=1)
    df = df.sample(n=7000)
    df[['author','formatted_prompt']].to_csv('./authorID.tsv',sep='\t',index=False) 