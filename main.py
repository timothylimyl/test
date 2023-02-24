"""
Sample from a trained model
"""
import os
import torch
import tiktoken
import streamlit as st
from PIL import Image
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
img = Image.open("images/st_logo.png")
st.set_page_config(
    page_title="ST-GPT",
    page_icon=img,
    layout="wide",
    initial_sidebar_state="expanded",
)

# remove streamlit menu and logo
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown('# :red[ST]:blue[GPT] :sunglasses:')
st.markdown("### This app generates RANDOM but coherent sentences using a language model")

st.markdown(":warning: Generation of text is not to be treated as factual")
st.markdown(":construction: On-going upgrades!")

st.markdown("\n\n\n\n\n")


# Fixed params
seed = 1337
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'

device_type = "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

# Set seed for deterministic output
torch.manual_seed(seed)

# Load model (Cache load)

@st.cache_resource
def load_model(path):
    ckpt_path = os.path.join(path, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


# Load tokeniser:
@st.cache_resource
def load_tokenisation():
    return tiktoken.get_encoding("gpt2")

# Cached
weights_path = os.getcwd()
model = load_model(path=weights_path)
enc = load_tokenisation()
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Allowing flexible user query
text_query = st.text_area("Input text (Press Ctrl+Enter once done):", value="Singapore Technologies (ST) Engineering is a company in Singapore that", height=250)
start_ids = encode(text_query)

num_samples = st.slider('Number of sentences generation', 1, 10, 1, step=1)
max_new_words = st.slider('Number of words in sentence output (approximated)', 1, 1000,200, step=10) 
max_new_tokens = int(max_new_words * 1.3333) # roughly 1 token = 3/4 of a word
temperature = st.slider('Randomness in Generation [Higher=More Random, Lower=Less Random, 1 = Neutral]', 0.0, 2.0, 0.7, step=0.1)
temperatre = temperature if temperature != 0 else 1e-6 #buggy due to multinomial with 0
# top_k = st.slider('Amount of output token', 1, 1000, 200, step=1)
top_k = 200

#emojify numbers
if st.button("Generate me some text!"):
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    # run generation
    with torch.no_grad():
        st.markdown("\n\n")
        for k in range(num_samples):
            with st.spinner("Generating (may take up to few minutes)....."):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                st.text_area(f":mailbox: Generation {k+1}", decode(y[0].tolist()), height=400)
                st.markdown("\n\n\n\n")

