from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel 

base_model = "meta-llama/Llama-3.2-1B"
adapter_model = "MESSItom/mathX-Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_model)
tokenizer = AutoTokenizer.from_pretrained(adapter_model)



alpaca_prompt = """Below is a Maths Question. Write an answer that appropriately completes the request.

### Question:
{}

### Answer:
"""

def generate_answer(sample: str, device,
                    max_new_tokens: int = 500):
    
    model = model.to(device=device)

    text = alpaca_prompt.format(sample)
    model_in = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    generation_output = model.generate(**model_in,
                                      max_new_tokens=max_new_tokens,
                                      temperature=0.7,
                                      eos_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)

    return generated_text.split("### Answer:\n")[1]
