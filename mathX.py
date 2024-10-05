from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, Latex

tokenizer = AutoTokenizer.from_pretrained("MESSItom/math-Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("MESSItom/math-Llama-3.2-1B")

alpaca_prompt = """Below is a Maths Question. Write an answer that appropriately completes the request.

### Question:
{}

### Answer:
"""

# Correct answer is 66
sample = """Let $A=\{1,2,3, \ldots \ldots \ldots \ldots, 100\}$. Let $R$ be a relation on A defined by $(x, y) \in R$ if and only if $2 x=3 y$. Let $R_1$ be a symmetric relation on $A$ such that $R \subset R_1$ and the number of elements in $R_1$ is n . Then, the minimum value of n is, $\qquad$"""
text = alpaca_prompt.format(sample)
display(Latex(text))

model_in = tokenizer(text, return_tensors="pt", truncation=True)
generation_output = model.generate(**model_in,
                                  max_new_tokens=200,
                                  temperature=0.7,
                                  eos_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
display(Latex(generated_text.split("### Answer:\n")[1]))