from transformers import MarianMTModel, MarianTokenizer

# Model name for English to Urdu translation
model_name = "Helsinki-NLP/opus-mt-en-ur"

# Loading tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text
def translate_en_to_ur(text):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode translation
    urdu_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return urdu_text

# Ask user for input
english_text = input("Enter a sentence in English: ")

# Translate and print
urdu_translation = translate_en_to_ur(english_text)

print("\nEnglish:", english_text)
print("Urdu:", urdu_translation)
