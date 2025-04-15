


import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langdetect import detect
import google.generativeai as genai

# Configure the Gemini API using environment variable
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class MultilingualTranslator:
    def __init__(self, requests_per_minute=1):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
        self.supported_languages = {
            'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
            'hi': 'Hindi', 'tr': 'Turkish', 'vi': 'Vietnamese', 'th': 'Thai'
        }
        # For rate limiting
        self.min_time_between_requests = 60.0 / requests_per_minute
        self.last_request_time = 0
        
    def detect_language(self, text):
        """Auto-detect the language of the input text"""
        try:
            lang_code = detect(text)
            language_name = self.supported_languages.get(lang_code, 'Unknown')
            return lang_code, language_name
        except:
            return 'unknown', 'Unknown'
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API quota errors"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_time_between_requests:
            # Sleep to respect rate limit
            sleep_time = self.min_time_between_requests - time_since_last_request
            print(f"Rate limiting: Waiting {sleep_time:.2f} seconds before next request...")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def translate(self, text, target_lang):
        """Translate text to the target language"""
        # Detect source language
        source_lang_code, source_lang_name = self.detect_language(text)
        
        # Get target language full name if only code provided
        if target_lang in self.supported_languages:
            target_lang_name = self.supported_languages[target_lang]
        else:
            # Assume target_lang is the full language name
            target_lang_name = target_lang
        
        # Create prompt for translation
        prompt = ChatPromptTemplate.from_template(
            """Translate the following text from {source_language} to {target_language}. 
            Provide only the translated text without any explanations or additional comments.
            
            Text to translate: {text}
            """
        )
        
        # Apply rate limiting before making the API call
        self._rate_limit()
        
        try:
            # Create a runnable sequence instead of LLMChain (new LangChain approach)
            chain = prompt | self.model
            
            # Invoke the chain
            response = chain.invoke({
                "source_language": source_lang_name,
                "target_language": target_lang_name,
                "text": text
            })
            
            translated_text = response.content
            
            return {
                "source_text": text,
                "source_language": source_lang_name,
                "target_language": target_lang_name,
                "translated_text": translated_text.strip()
            }
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return {
                "source_text": text,
                "source_language": source_lang_name,
                "target_language": target_lang_name,
                "translated_text": f"Error: {str(e)}",
                "error": True
            }
    
    def get_supported_languages(self):
        """Return a list of supported languages"""
        return self.supported_languages
    
    def display_supported_languages(self):
        """Display all supported languages in a readable format"""
        print("\nSupported Languages:")
        print("--------------------")
        for code, name in sorted(self.supported_languages.items(), key=lambda x: x[1]):
            print(f"{code}: {name}")


def get_user_input():
    """Get translation inputs from the user"""
    print("\n===== AI-Powered Language Translator =====\n")
    
    # Initialize translator
    translator = MultilingualTranslator(requests_per_minute=1)
    
    # Show supported languages
    translator.display_supported_languages()
    
    while True:
        print("\n")
        print("1. Translate text")
        print("2. Exit")
        choice = input("\nSelect an option (1-2): ")
        
        if choice == '2':
            print("Thank you for using the translator. Goodbye!")
            break
        
        elif choice == '1':
            # Get text to translate
            text = input("\nEnter the text to translate: ")
            if not text.strip():
                print("Text cannot be empty. Please try again.")
                continue
            
            # Get target language
            print("\nEnter target language code or name (e.g., 'es' or 'Spanish')")
            target_lang = input("Target language: ")
            if not target_lang.strip():
                print("Target language cannot be empty. Please try again.")
                continue
            
            # Perform translation
            print("\nTranslating...")
            result = translator.translate(text, target_lang)
            
            if result.get("error", False):
                print("An error occurred during translation.")
            else:
                print("\nTranslation Result:")
                print("-----------------")
                print(f"Source ({result['source_language']}): {result['source_text']}")
                print(f"Translation ({result['target_language']}): {result['translated_text']}")
        
        else:
            print("Invalid option. Please try again.")


# Main execution
if __name__ == "__main__":
    get_user_input()