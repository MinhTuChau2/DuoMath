import tkinter as tk
from tkinter import Button
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import random

# Initialize the model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Create the Tkinter window
root = tk.Tk()
root.title("Handwriting Recognition")

# Set up the canvas for drawing
canvas_width = 400
canvas_height = 200
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# This will hold the drawn image
image = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(image)

# Draw line on the canvas
def draw_on_canvas(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=2)
    draw.line([x1, y1, x2, y2], fill="black", width=2)

canvas.bind("<B1-Motion>", draw_on_canvas)

# Function to generate a new math question

def generate_math_question():
    global correct_answer
    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)
    operation = random.choice(["+", "-"])
    
    # If the operation is subtraction, make sure the larger number is first
    if operation == "-":
        if num1 < num2:
            num1, num2 = num2, num1  # Swap to make sure num1 >= num2
    
    # Create the math question and the correct answer
    question_text = f"{num1} {operation} {num2} = ?"
    correct_answer = str(eval(f"{num1} {operation} {num2}"))
    
    # Update the question label
    math_question_label.config(text=question_text)


# Function to process the drawn image and run the model
def recognize_text():
    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Generate predictions
    generated_ids = model.generate(pixel_values)
    
    # Decode the predicted text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Clean up the generated text
    cleaned_text = ' '.join(generated_text.split())  # Remove extra spaces
    
    # Extract only the last number (assuming the second number is correct)
    cleaned_text_split = cleaned_text.split()
    if len(cleaned_text_split) > 1:
        cleaned_text = cleaned_text_split[-1]  # Take the last number
    else:
        cleaned_text = cleaned_text_split[0]  # If only one number, keep it
    
    # Show the cleaned recognized text in a label
    result_label.config(text="Detected Text: " + cleaned_text)
    
    # Check if the answer is correct
    if cleaned_text.strip() == correct_answer:
        update_score(1)

# Function to update the score and generate a new question
def update_score(points):
    global score
    score += points
    score_label.config(text=f"Score: {score}")
    
    if points == 1:
        clear_canvas()
    # Generate a new math question when the score increases
    generate_math_question()

# Function for the skip button, generates a new math question
def skip_question():
    generate_math_question()

# Button to trigger text recognition
recognize_button = Button(root, text="Recognize Text", command=recognize_text)
recognize_button.pack()

# Label to display the recognized text
result_label = tk.Label(root, text="Detected Text: ")
result_label.pack()

# Clear button to reset the canvas
def clear_canvas():
    canvas.delete("all")
    # Reset the image to all white
    global image, draw
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)

clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Skip button to generate a new question without checking the answer
skip_button = Button(root, text="Skip", command=skip_question)
skip_button.pack()

# Label to display the math question
math_question_label = tk.Label(root, text="Math Question will appear here")
math_question_label.pack()

# Initialize score
score = 0
score_label = tk.Label(root, text=f"Score: {score}")
score_label.pack()

# Generate the first math question
generate_math_question()

# Run the Tkinter loop
root.mainloop()
