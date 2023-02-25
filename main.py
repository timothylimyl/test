import os
import torch
import tiktoken
import streamlit as st
from PIL import Image
from model import GPTConfig, GPT
import gdown
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
def load_model(URL):

    # Weights download from gdrive
    weights_path = "ckpt.pt"
    if not os.path.exists(weights_path):
        gdown.download(URL,output=weights_path,quiet=False)

    checkpoint = torch.load(weights_path, map_location=device)
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
model = load_model(URL="https://drive.google.com/uc?id=14I94WUIMtE5rm_wVAp3bO-TglPgCfRNX")
enc = load_tokenisation()
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Allowing flexible user query
def_input_text = "Singapore Technologies (ST) Engineering is a company in Singapore that"
def_new_words = 200
def_temperature = 0.7

text_query = st.text_area("Input text (Press Ctrl+Enter once done):", value=def_input_text, height=250)
start_ids = encode(text_query)

num_samples = st.slider('Number of sentences generation', 1, 5, 1, step=1)
max_new_words = st.slider('Number of words in sentence output (approximated)', 0, 1000,def_new_words, step=10) 
max_new_tokens = int(max_new_words * 1.3333) # roughly 1 token = 3/4 of a word
temperature = st.slider('Randomness in Generation [Higher=More Random, Lower=Less Random, 1 = Neutral]', 0.0, 2.0, def_temperature, step=0.1)
temperature = temperature if temperature != 0.0 else 1e-6 #buggy due to multinomial with 0
# top_k = st.slider('Amount of output token', 1, 1000, 200, step=1)
top_k = 200

# default output to default input
defaults = [
    """
    Singapore Technologies (ST) Engineering is a company in Singapore that focuses on advancing the satellite’s, communications, and Internet of Things (IoT) ecosystem. The Group Engineering Structure is comprised of the following: (a) The Defence Business Council (DBC), comprising Richard Thorne, (b) Peter Tan, (c) Ng Sing Chan and (d) Ng Hong Kong Poh Lian), who together with David Kwok Moon of ST Engineering have been responsible for shaping the company’s culture since its launch, and spearhead the ST Engineering ethos since its launch.

From humble beginnings as a commercial airframe maintenance and repair company (MQO) until its founding in 1967 as the subsidiary to the Royal Thai Navy (RAW), the ST Engineering ethos has remained the same. Advised by visionary leaders and fuelled by abundant conviction, persistence and determination, the ST Engineering ethos has made every decision through strict conformity to the code of conduct.

The code mandates that no later than the designated time when the aircraft is not safe to fly or touch, no further action is taken. The Group Executive Committee (EXCO) is accountable to the higher echelons of the nouveau-riche the ST Engineering is, ensuring that responsibility for the organisation’s success is exercised prudently.

In keeping to the code
    """,

    """
    Singapore Technologies (ST) Engineering is a company in Singapore that focuses on advancing the satellite ® ground network,” said Mr Lim Serh Ghee, the Chairman and CEO of ST Engineering. “The establishment of this important partnership with ST will facilitate the development of deep technical capabilities,” he said. 

RAINMAKERS OF CHANGE S T Engineering is a global technology, defence and engineering group with a diverse portfolio of businesses across the aerospace, smart city, defence and public security segments. The Group has commercial aircraft and engine MRO operations in its network of more than 150 subsidiaries in more than 150 countries, with a global network expanding at a steady clip. With a workforce of over 2,000 at ST Engineering’s global headquarters in Tianjin, China, its parent company Singapore Air Transport Systems Company, (STST.S) and ST Engineering's (STEC) subsidiaries in Asia, Europe, the Middle East and the U.S., the U.K. and the U.S., the Group has strong business fundamentals and positive long-term growth prospects. It has helped to drive global enterprise growth by deepening its focus into new markets and by new market segments. For instance, its new Flexibility Zone in 13 continents will see foreign air cargo growth in the next few years, it said. In our opinion, the Group
    """,

    """

Singapore Technologies (ST) Engineering is a company in Singapore that focuses on advancing technology-enabled smart cities through its extensive suite of products and services for smart lighting, air-conditioning and security applications. For its ongoing credit ratings, SustainabilityRevised is recommended by SFA.

SustainabilityRevised.pdf (4.0 MB, PDF) is a data link tool that graphs and analyses the data provided by the three listed companies to provide a visual overview of their key business performance and key trends. It is designed to be accurate, highlighting key technology innovations and key entities that contributed to the industry’s journey forward,” said Chye Cing. “The data provided in the download form is valuable in understanding how the industry has moved towards more dependable and sustainable operating models, and in helping the industry to establish sustainable business models.”

The data provided in the R&D and innovation centre are set to be a critical part of the ST Engineering’s business journey, helping the company to establish a global customer base and achieve its corporate ambitions.

As the battle for resource resources in Singapore continues to be shaped by deep rooted land transportation and a desire to protect the environment, it is imperative that ST Engineering can build on the strength and versatility of its products and services based on needs and demand. 
    
    """,


    """
Singapore Technologies (ST) Engineering is a company in Singapore that is the main commercial joint venture (COB) for ST Engineering today.

Singapore Technologies Engineering (ST Engineering), the world’s largest commercial airframe MRO company by maintenance manhours, has about as much landside and airside operations as any other company, with 10,000 engineers and 11,000 staff. ST Engineering is headquartered in Singapore, and its headquarters are in cities all over the world.

Its main office is in the city of Singapore, but its operations are mainly spread out in North America. Its headquarters are in 1st district of Changi, China, where work is being done by Sun Microsystems (SINA) and Siemens Corp (SINA)

1. What is ST Engineering?

ST Engineering is a global technology, defense and engineering group specialising in the aerospace, electronics, land systems and marine sectors.

Its core purpose is to contribute to the security and sustainability of a world-leading performanceally and linearly efficient R&D capacity and resources to Singapore’s Ministry of Defence (MINDEF) and MINN, the national MINDEF hospital.

Its sole shareholder is Singapore's GICOS, a.k.a. ST Engineering Corp.
    """,

    """
   Singapore Technologies (ST) Engineering is a company in Singapore that focuses on advancing the technology of aerospace wheel engineering from a humble workshop to a global flagship aerospace engineering centre. We believe that a strong engineering culture can provide the foundation for a strong business and engineering nation, where subsystems are designed to deter attacks from multiple angles, wheels – the engine of aerospace operations – are never far from the technicians to ensure mission success. We were selected by Virgin Australia to design and build the first-ever airframe MRO platform to provide the ground transportation infrastructure for the development of the next phase of ST Aerospace’s airframe MRO business. As part of our corporate mission to save lives and property, our MRO work was further supported by passenger-to-freighter conversions by ST Aerospace. The success of this programme has been a lesson for the future of airframe MRO.”, “Achieving sustainability in airframe MRO is as about managing technology and innovation,” said Mr Quek Poh Huat, President & CEO of ST Engineering. “ MRO is increasingly deployed to by-the-source MRO solutions for complex systems that could include the collection of data of benefits ranging from health and environmental impact to economic, psychological, and environmental impact.  
    """
]

if st.button("Generate me some text!"):
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    # run generation

    # set a default input and output for quick response:
    if text_query == def_input_text and max_new_words == def_new_words and temperature == def_temperature:       
        st.markdown("Pre-generated \n\n")
        for i in range(min(num_samples,len(defaults))):
            st.text_area(f":mailbox: Generation {i+1}", defaults[i], height=400)
            st.markdown("\n\n\n\n")
    else: # if not default then generate for user
        with torch.no_grad():
            st.markdown("\n\n")
            for k in range(num_samples):
                with st.spinner("Generating (may take up to few minutes)....."):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    st.text_area(f":mailbox: Generation {k+1}", decode(y[0].tolist()), height=400)
                    st.markdown("\n\n\n\n")

