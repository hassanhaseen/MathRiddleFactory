import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Basic page setup
st.set_page_config(
    page_title="Math Riddle Factory",
    page_icon="🧮",
    layout="centered"
)

# Title & Description
st.title("🧠 Math Riddle Factory")
st.caption("Generate creative math riddles using AI!\n")
st.markdown("---")

# Load model and tokenizer from HuggingFace Hub
@st.cache_resource
def load_model():
    with st.spinner("Loading Math Riddle GPT-2 Model... Please wait."):
        tokenizer = GPT2Tokenizer.from_pretrained("hassanhaseen/MathRiddleGPT2")
        model = GPT2LMHeadModel.from_pretrained("hassanhaseen/MathRiddleGPT2")
    return tokenizer, model

tokenizer, model = load_model()

# Sidebar with controls
st.sidebar.header("🔧 Controls")
num_riddles = st.sidebar.slider("Number of riddles", 1, 5, 3)
temperature = st.sidebar.slider("Temperature (creativity)", 0.1, 1.0, 0.7)

# Generate riddles button
generate_btn = st.button("✨ Generate Riddles!")

if generate_btn:
    with st.spinner("Generating riddles... Give me a sec! 🤖"):
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

        time.sleep(1)  # just to show spinner for a sec :)

    st.markdown("---")
    st.subheader("📝 Your Math Riddles")

    # Display riddles one by one with click-to-reveal answers
    for i, output in enumerate(outputs):
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Separate Riddle and Answer
        try:
            riddle_part, answer_part = full_text.split("Answer:")
        except ValueError:
            riddle_part = full_text
            answer_part = "Oops! Couldn't find the answer."

  # Display riddle
        with st.expander(f"❓ Riddle {i+1}: {riddle_part.strip()}"):
            st.success(f"✅ Answer: {answer_part.strip()}")



st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888888;
        }
        .footer span {
            position: relative;
            cursor: pointer;
            color: #FF4B4B;
        }
        .footer span::after {
            content: "Hassan Haseen & Sameen Muzaffar";
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: #fff;
            padding: 5px 10px;
            border-radius: 8px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
            font-size: 12px;
        }
        .footer span:hover::after {
            opacity: 1;
        }
    </style>

    <div class='footer'>
        Created with ❤️ by <span>Team CodeRunners</span>
    </div>
    """,
    unsafe_allow_html=True
)
