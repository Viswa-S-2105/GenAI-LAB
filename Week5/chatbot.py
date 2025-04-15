
# ✅ Set your API key directly
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyBy7DKmbxQt2m1AgdQ93Ja8ciJvvt7dtwM"

# ✅ Import Gemini API
import google.generativeai as genai

# ✅ Configure the Gemini client
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Load the Gemini 1.5 Pro model
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# ✅ Start chat session
chat = model.start_chat()

# ✅ Chat loop
print("🤖 Gemini 1.5 Chatbot is ready! Type 'exit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Bot: Goodbye! 👋")
        break
    response = chat.send_message(user_input)
    print("Bot:", response.text)
