from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from PIL import Image
import streamlit as st
from languages import languages

st.set_page_config(layout='wide')

image = Image.open('banner.png')
st.image(image,use_column_width='auto')
st.sidebar.header('🌟 Please choose the languages 🗣')

src_lang = st.sidebar.selectbox('Select source language for Translation 🎯', list(languages.keys()), key = 1)
trans_lang = st.sidebar.selectbox('Select target language for translation 🎯', list(languages.keys()), key = 2)

src_lang_code = [value for key, value in languages.items() if key == src_lang]

trans_lang_code = [value for key, value in languages.items() if key == trans_lang]

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def download_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    return model, tokenizer

st.title('✨ Multilingual Translator 🗣')
text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)
model, tokenizer = download_model()

if st.button('💬 Translate'):
    if text == '':
        st.warning('⚠ Please enter the required text for translation! 😦')
    else:
        with st.spinner(text='Translating...💫'):
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            tokenizer.src_lang = str(src_lang_code[0])
            encoded_hindi_text = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_hindi_text, forced_bos_token_id=tokenizer.lang_code_to_id[str(trans_lang_code[0])])
            out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            st.success('✅ Translation Complete! 😉')
            st.write('', str(out).strip('][\''))
else: pass

st.markdown("<br><br><hr><center>Made with ❤️ by <a href='mailto:ralhanprateek@gmail.com?subject=Multilingual Translator WebApp!&body=Please specify the issue you are facing with the app.'><strong>Prateek Ralhan</strong></a></center><hr>", unsafe_allow_html=True)
