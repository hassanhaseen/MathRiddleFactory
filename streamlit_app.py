import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Title and Description
st.title("🧠 Math Riddle Factory")
st.subheader("Generate fun, creative math riddles with AI!")

# Load model and tokenizer from Hugging Face Hub
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("hassanhaseen/MathRiddleGPT2")
    model = GPT2LMHeadModel.from_pretrained("hassanhaseen/MathRiddleGPT2")
    return tokenizer, model

tokenizer, model = load_model()

# User input controls
num_riddles = st.slider("How many riddles do you want to generate?", 1, 5, 3)
temperature = st.slider("Temperature (controls creativity)", 0.1, 1.0, 0.7)

# Generate button
if st.button("Generate Riddles!"):
    prompt = "<|startoftext|>Riddle:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=num_riddles,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=temperature
    )

    st.subheader("Here are your riddles!")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        st.markdown(f"**Riddle {i+1}:** {text}")

# Footer
st.caption("Made by Hassan Haseen | Roll No: 21F-9221")
