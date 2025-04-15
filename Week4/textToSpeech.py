import tkinter as tk
from tkinter import messagebox
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_text():
    text = text_input.get("1.0", tk.END).strip()
    if text == "":
        messagebox.showwarning("Input Needed", "Please enter some text.")
    else:
        engine.setProperty('rate', 150)    # Speech speed
        engine.setProperty('volume', 1.0)  # Volume level
        engine.say(text)
        engine.runAndWait()

# Create GUI window
root = tk.Tk()
root.title("Text to Speech Converter")
root.geometry("400x300")
root.config(bg="#f0f0f0")

# App Title
title_label = tk.Label(root, text="Text to Speech (Offline)", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
title_label.pack(pady=10)

# Text Input Area
text_input = tk.Text(root, height=8, width=40, font=("Arial", 12))
text_input.pack(pady=10)

# Speak Button
speak_button = tk.Button(root, text="🔊 Speak", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=speak_text)
speak_button.pack(pady=10)

# Run the GUI loop
root.mainloop()
