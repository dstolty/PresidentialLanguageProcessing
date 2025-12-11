Instructions for running code:

**DISCLAIMER** I cannot add the tensor files that hold the LoRA model weights to github since they are too big. If needed, I can provide them.

CSV files contains the promtps used for training the model. TSV files contain the original data pulled from the internet. TXT files contain author lists or responses from the models after training. 

**Similar to the model weights, I cannot include the CSVs since they are too large. I have included sample TSVs which contain a subset of our dataset. These can be used in conjuction with our scripts to create the prompts.

For scraper.py:

You must know how many pages are in the category you want to target. Put that number in the for loop. Set the target url to desired category on UCSB Presidency Project. Make sure your output directory is chosen. Run script and the scraper will automatically populate a tsv file with columns Author, Date, Text

## Gemini used to create most of this script

LoRA.py

Set output directory, set base model, choose hyperparamters. Run trainer. Will take a long time.

## Gemini used to help create this script

createPrompts.py

Small script used to transform the dataset into a collection of prompts for training the model. Code to create generative prompts commented out. Currently formatted to create classification promtps. Ensure input and output files are correct before running.

testingIDprompts.py

Near mirror of above script minus the response part. Used to create the prompts for testing author identifcation. Randomly samples 7000 records from the spoken records dataset

identifcation.py 

Failed script. Ensure input file is correct. Ensure correct LoRA checkpoint is loaded (will be in ID_Lora folder for this task). Run script. Model will output to terminal and the answers will not make that much sense. There will be a few that have names but most won't.

## Gemini used to create this script

chatinterface.py 

A child of Gemini and our prior lab. This is the interface to generate output from our trained models. Ensure correct LoRA checkpoint is loaded, either EO_lora or speech_lora (EO trained on both EO and speeches, while speech just on speeches). Ensure base model is LLama-7b. Adjust hyperparamters if necessary. 

