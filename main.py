from MathX import generate_answer, get_model
from torch import cuda
from dotenv import load_dotenv
import os

load_dotenv()

from PIL import Image
from pix2tex.cli import LatexOCR

# device = "cuda" if cuda.is_available() else "cpu"
# model, tokenizer = get_model(device="cuda")

ocr = LatexOCR()
img = Image.open("image.png")

print(ocr(img))