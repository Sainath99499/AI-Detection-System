from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================================
# LOAD MODEL
# =========================================

print("Loading explanation engine...")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Explanation engine loaded!")

# =========================================
# GENERATE EXPLANATION
# =========================================

def generate_explanation(
    prediction,
    ai_probability
):

    prompt = f"""
    Explain why content may be classified as
    {prediction} with AI probability
    of {ai_probability}%.
    Keep explanation short and simple.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_length=80
    )

    explanation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return explanation

# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    response = generate_explanation(
        "AI Generated Text",
        87
    )

    print(response)