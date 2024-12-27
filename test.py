from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-sla"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Function to translate text
def translate(text):
    # Skip empty strings
    if not text.strip():
        return ""

    # Preprocessing: Remove extra whitespace and normalize punctuation
    text = text.strip()
    
    # Tokenize with more careful padding and attention mask
    inputs = tokenizer(
        text, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Set explicit input length limit
        return_attention_mask=True
    )

    # Enhanced translation parameters
    translated = model.generate(
        **inputs,
        max_length=100,       # Increased max length for better completeness
        min_length=5,         # Prevent too short translations
        temperature=0.6,      # Slightly reduced temperature for more focused output
        top_k=20,            # Reduced top_k for more precise word choice
        num_beams=5,         # Add beam search for better translation quality
        early_stopping=True,  # Stop when valid translation is found
        no_repeat_ngram_size=2,  # Prevent repetition
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example usage
input_text = "Yes, please, baby. Finger me harder while you play with my nipples"

# split on dot then translate then merge
input_text = input_text.split(".")
translated_text = [translate(text) for text in input_text]
print(" ".join(translated_text))
