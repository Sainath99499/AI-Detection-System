from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = None
model = None

# =========================================
# GENERATE EXPLANATION
# =========================================

def generate_explanation(
    prediction,
    ai_probability
):

    global tokenizer, model

    if tokenizer is None or model is None:
        print("Loading explanation engine...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )
        print("Explanation engine loaded!")

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