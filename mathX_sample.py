from IPython.display import display, Latex
from MathX import generate_answer

# Correct answer is 66
sample = """Let $A=\\{1,2,3, \\ldots \\ldots \\ldots \\ldots, 100\\}$. Let $R$ be a relation on A defined by $(x, y) \\in R$ if and only if $2 x=3 y$. Let $R_1$ be a symmetric relation on $A$ such that $R \\subset R_1$ and the number of elements in $R_1$ is n . Then, the minimum value of n is, $\\qquad$"""
answer = generate_answer(sample, device="cuda")

print('\n'*5)
print("Question:\n\n")
print(display(Latex(sample)))
print("Answer:\n\n")
print(display(Latex(answer)))