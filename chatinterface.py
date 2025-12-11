import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
# Your base Llama 7B model ID (use the correct one you fine-tuned from)
BASE_MODEL_ID = "/SI425/Llama-2-7b-hf" 
# The local path where your LoRA adapter weights are saved
LORA_ADAPTER_PATH = "./EO_lora" 
# How the model should respond to your input (important for chat models)
SYSTEM_PROMPT = "You have been trained on thousands of executive orders. You know how to write one and should respond with an executive order."
# Max length of the response the model generates
MAX_NEW_TOKENS = 512 
# Use 8-bit quantization for lower VRAM usage (requires bitsandbytes installed)
LOAD_IN_8BIT = False 
# --- End Configuration ---

def load_model_and_tokenizer(base_model_id, lora_adapter_path, load_in_8bit=False):
    """Loads the base model, applies the LoRA adapter, and loads the tokenizer."""
    print("Loading base model and tokenizer...")
    
    # Determine the device and create the device_map for model loading
    if torch.cuda.is_available():
        # Use 'cuda' for the device name
        device = "cuda" 
        # Use the device_map dict for AutoModelForCausalLM loading
        device_map = {"": 0} 
    else:
        # Fallback to 'cpu' or 'auto'
        device = "cpu"
        device_map = "auto"

    # Load Tokenizer (unchanged)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token 

    # --- Loading Base Model ---
    model_args = {
        "device_map": device_map, # <-- This takes the device_map dict or string
        "torch_dtype": torch.float16,
    }
    # ... (Quantization setup omitted for brevity but stays the same) ...

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **model_args,
    )

    # Load and Merge LoRA Adapter (unchanged)
    #odel = PeftModel.from_pretrained(model, lora_adapter_path)
    #odel = model.merge_and_unload()
    #model.eval()
    
    # --- The Fix is HERE ---
    # When you call .to(), you pass the single device name, NOT the dictionary.
    if torch.cuda.is_available() and device_map == "auto":
        # Only move the model explicitly if we didn't use a full device_map 
        # in from_pretrained (e.g., if using CPU or a complex distributed setup).
        # When device_map="auto" or a dict is used, the model is already on the device.
        pass # The model is already placed by device_map="auto"

    print("✅ Model and adapter loaded successfully. Ready to chat.")
    # Return the model and tokenizer, which are already placed on the correct device.
    return model, tokenizer

def create_prompt(system_prompt, user_input):
    """
    Creates a prompt using the Llama-style instruction format.
    You might need to adjust this if your fine-tuning used a different template.
    """
    return f"<s> [INST] {user_input} [/INST] Remarks: "

def generate_response(model, tokenizer, prompt, max_new_tokens):
    """Generates and decodes the model's response."""
    # 1. Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 2. Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

    # 3. Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Use sampling for more creative/chat-like responses
            top_k=50,
            top_p=0.95,
            temperature=0.7 # Adjust temperature for creativity/predictability
        )
    
    # 4. Decode the generated tokens
    # Decode only the newly generated text (after the input prompt)
    output_text = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):], 
        skip_special_tokens=True
    ).strip()
    
    return output_text

def main():
    """Main function for the interactive chat loop."""
    try:
        model, tokenizer = load_model_and_tokenizer(
            BASE_MODEL_ID, LORA_ADAPTER_PATH, LOAD_IN_8BIT
        )
        
        print("\n--- Start Chat ---")
        print(f"System: {SYSTEM_PROMPT}")
        print("Type 'quit' or 'exit' to end the session.")
        
        while True:
            # Get user input
            user_input = input("\n> User: ")
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit"]:
                print("\nSession ended. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Create the formatted prompt
            prompt = create_prompt(SYSTEM_PROMPT, user_input)
            
            # Generate and print the response
            print("\n> Model: ", end="", flush=True)
            response = generate_response(model, tokenizer, prompt, MAX_NEW_TOKENS)
            print(response)

    except FileNotFoundError:
        print(f"🚨 Error: Could not find model or adapter at the specified paths.")
        print("Please check BASE_MODEL_ID and LORA_ADAPTER_PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()