from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import os

app = Flask(__name__)

# Point to the directory containing the standard HF model files
local_hf_model_dir = "/Mistral-7B-Instruct-v0.1"

# Check if the directory exists
if not os.path.exists(local_hf_model_dir) or not os.path.isdir(local_hf_model_dir):
    print(f"Error: Hugging Face model directory not found at {local_hf_model_dir}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = None
model = None
try:
    # Load tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(local_hf_model_dir, trust_remote_code=True) # remote code is a security concern
    model = AutoModelForCausalLM.from_pretrained(
        local_hf_model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Use float16 on GPU
        device_map="auto", # Automatically maps model to available devices
        trust_remote_code=True
    )
    print("Model and tokenizer loaded successfully from local Hugging Face directory!")

except Exception as e:
    print(f"Error loading model from local Hugging Face directory: {e}")
    # Ensure model and tokenizer are None if loading failed
    tokenizer = None
    model = None

def custom_generate_text(model, input_ids: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
    # `past_key_values` should be initialized to `None` for the first call
    # It will be populated by the model's output in subsequent iterations.
    past_key_values = None

    # Determine device from model's weights
    device = model.model.embed_tokens.weight.device
    
    # Ensure input_ids are on the correct device
    input_ids = input_ids.to(device)
    
    origin_len = input_ids.shape[-1] # Length of the initial prompt tokens
    output_ids = input_ids.clone()
    next_token_input = input_ids # For the first step, feed the whole input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # If past_key_values exist, we only pass the most recent token as input_ids
            # This is crucial for efficient auto-regressive decoding.
            current_input_for_model = next_token_input if past_key_values else input_ids

            out = model(
                input_ids=current_input_for_model,
                past_key_values=past_key_values,
                use_cache=True  # Important to enable caching of KV values
            )
            
            # Get logits for the last token in the sequence
            logits = out.logits[:, -1, :]
            
            # Greedy decoding: pick the token with the highest logit
            token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append the newly generated token to the output sequence
            output_ids = torch.cat([output_ids, token], dim=-1)
            
            # Update past_key_values for the next iteration
            past_key_values = out.past_key_values
            
            # The next input to the model will be just the newly generated token
            next_token_input = token.to(device)

            # Check for EOS token
            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break

    # Return just the newly generated part (excluding the original prompt)
    return output_ids[:, origin_len:]

# --- Flask Routes ---
@app.route('/generate', methods=['POST'])
def generate_route():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not loaded. Server is not ready."}), 503 # Service Unavailable

    data = request.get_json(force=True)
    prompt = data.get('prompt', "Write a short story.")
    max_new_tokens = data.get('max_new_tokens', 50)
    
    # Ensure max_new_tokens is a positive integer
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        max_new_tokens = 50 # Default if invalid

    try:
        # Prepare input for the model
        # Mistral-instruct uses a chat template, so apply it for consistent formatting
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        # Ensure input_ids are on the same device as the model.
        # The custom_generate_text function also handles this, but good to be explicit.
        input_ids = input_ids.to(device) 

        # Call your custom generation function
        generated_token_ids = custom_generate_text(model, input_ids, max_new_tokens=max_new_tokens)
        
        # Decode the generated token IDs back to text
        generated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})
    except Exception as e:
        # Log the full exception for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error during text generation: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
