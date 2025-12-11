import json
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

if __name__ == '__main__':
## THe following is modified code from GEmini 
## All heavily modified except for the create_prompt function (slightly modified)
    ID = "/SI425/Llama-2-7b-hf"

    LORA_PATH = "./ID_lora"

    # Define quantization settings (4-bit) to reduce model size when loaded.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                  # enable 4-bit quantization
        bnb_4bit_quant_type="nf4",          # use "normal float 4" quantization
        bnb_4bit_compute_dtype=torch.float16,  # compute in half precision
    )

    model = AutoModelForCausalLM.from_pretrained(
        ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically uses CUDA if available
        trust_remote_code=True
    )
    model.config.use_cache = False # Recommended for training

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ID, trust_remote_code=True)
    # Set a padding token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fixes warning

    # LoRA Hyperparameters
    LORA_R = 16          # The rank (r) of the update matrices, 16 default
    LORA_ALPHA = 32      # The scaling factor (alpha), 32 deafult ( 32/16 = 2 scale factor)
    LORA_DROPOUT = 0.05 ## 0.05 default
    # Target modules are the layers to apply LoRA to (e.g., attention projections)
    # For Llama models, 'q_proj', 'k_proj', 'v_proj', 'o_proj' are common targets.
    LORA_TARGET_MODULES = [ ## target all layers, not necessary but most robust ( for last iteration of training for author ID we only trained the self attention layers)
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj", ## self attention layers ^^
    ] ## add gate_proj, up_proj, down_proj for feed forward layers

    ## only necessary for the first LoRA iteration, will be drawn from the adapter files in subsequent LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS", # For generative models
    )



    training_args = SFTConfig(
        output_dir="./ID_lora",
        num_train_epochs=3, ## default 3 epochs
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, # Increase for effective larger batch size (4*4=16)
        optim="paged_adamw_8bit", # Optimizer optimized for QLoRA
        logging_steps=100, ## every 100
        learning_rate=2e-4, ## general practice learning rate
        fp16=True,
        save_strategy="epoch", ## save model weights after each epoch
        dataset_text_field='formatted_prompt', ## specify which field of dataset to use
        packing=False,
        max_length=1024) ## lower number lessens memory requirement

    df = pd.read_csv('classification_prompts.csv')
    formatted_data = Dataset.from_pandas(df) ## SFTTrainer expects dataset object, not pd df

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_data,
        peft_config=lora_config, 
        processing_class=tokenizer,
        args=training_args) ## apply LORA config, send training args, send data to model

    trainer.train()

    trainer.model.save_pretrained(training_args.output_dir)
