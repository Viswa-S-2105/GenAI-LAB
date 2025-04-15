import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

# Predefined language pairs and models
lang_pairs = {
    "English to French": ("en", "fr"),
    "French to English": ("fr", "en"),
    "English to Hindi": ("en", "hi"),
    "Hindi to English": ("hi", "en"),
    "English to Tamil": ("en", "ta"),
    "Tamil to English": ("ta", "en"),
    "English to Spanish": ("en", "es"),
    "Spanish to English": ("es", "en")
}

def get_model(source_lang, target_lang):
    # Ensure valid language pair
    if f"{source_lang} to {target_lang}" not in lang_pairs:
        raise ValueError(f"Invalid language pair: {source_lang} to {target_lang}")
    
    # Construct the model name
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    print(f"Loading model: {model_name}")  # Debugging line to check model name

    try:
        # Load the model and tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"
    
    return model, tokenizer

def translate(text, lang_pair):
    # Extract source and target language from the selected pair
    source_lang, target_lang = lang_pairs[lang_pair]
    
    # Get the model and tokenizer for the selected languages
    model, tokenizer = get_model(source_lang, target_lang)
    
    # Handle error if model loading fails
    if model is None:
        return tokenizer  # In case of an error, return the error message
    
    # Tokenize the input text and translate
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Gradio Interface with language selector dropdowns
interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(lines=4, label="Enter Text"),
        gr.Dropdown(choices=list(lang_pairs.keys()), label="Select Language Pair", value="English to French")
    ],
    outputs="text",
    title="Dynamic Translator",
    description="Translate between various languages using MarianMT models."
)

interface.launch() 