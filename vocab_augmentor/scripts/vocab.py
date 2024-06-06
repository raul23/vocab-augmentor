import logging
import os
import string
import subprocess
import warnings

# Suppress the specific warning by setting the logging level to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

import google.generativeai as genai
import jieba
import jieba.posseg as pseg
import langcodes
import pandas as pd
import spacy
import torch
import transformers

from huggingface_hub import notebook_login
from pypinyin import pinyin, Style
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline

api_token = os.environ.get('GEMINI_API_TOKEN')

# Define a mapping for POS tags to their full names
POS_MAP = {
    'a': 'adjective',
    'ad': 'adverbial',
    'ag': 'adjective morpheme',
    'an': 'nominal adjective',
    'b': 'distinguishing word',
    'c': 'conjunction',
    'd': 'adverb',
    'e': 'exclamation',
    'f': 'directional noun',
    'g': 'root',
    'h': 'prefix',
    'i': 'idiom',
    'j': 'abbreviation',
    'k': 'suffix',
    'l': 'temporal noun',
    'm': 'numeral',
    'n': 'general noun',
    'ng': 'noun morpheme',
    'nr': 'person name',
    'ns': 'place name',
    'nt': 'organization name',
    'nz': 'other proper noun',
    'o': 'onomatopoeia',
    'p': 'preposition',
    'q': 'quantity',
    'r': 'pronoun',
    's': 'space',
    't': 'time',
    'tg': 'time morpheme',
    'u': 'auxiliary',
    'ul': 'particle',
    'v': 'verb',
    'vd': 'adverbial verb',
    'vg': 'verb morpheme',
    'vn': 'nominal verb',
    'w': 'punctuation',
    'x': 'non-morpheme character',
    'y': 'modal particle',
    'z': 'status word',
    'un': 'unknown'
}

# Common Chinese particles to handle
common_particles = {
    "的": "of",
    "了": "le (completed action marker)",
    "是": "is/am/are",
    "在": "at/in",
    "有": "have/has",
    "和": "and",
    "吗": "ma (question particle)",
    "不": "not",
    "我": "I/me",
    "你": "you",
    "他": "he/him",
    "她": "she/her",
    "它": "it",
}

facebook_languages = {
  'ar': 'ar_AR',
  'cs': 'cs_CZ',
  'de': 'de_DE',
  'en': 'en_XX',
  'es': 'es_XX',
  'et': 'et_EE',
  'fi': 'fi_FI',
  'fr': 'fr_XX',
  'gu': 'gu_IN',
  'hi': 'hi_IN',
  'it': 'it_IT',
  'ja': 'ja_XX',
  'kk': 'kk_KZ',
  'ko': 'ko_KR',
  'lt': 'lt_LT',
  'lv': 'lv_LV',
  'my': 'my_MM',
  'ne': 'ne_NP',
  'nl': 'nl_XX',
  'ro': 'ro_RO',
  'ru': 'ru_RU',
  'si': 'si_LK',
  'tr': 'tr_TR',
  'vi': 'vi_VN',
  'zh': 'zh_CN',
  'af': 'af_ZA',
  'az': 'az_AZ',
  'bn': 'bn_IN',
  'fa': 'fa_IR',
  'he': 'he_IL',
  'hr': 'hr_HR',
  'id': 'id_ID',
  'ka': 'ka_GE',
  'km': 'km_KH',
  'mk': 'mk_MK',
  'ml': 'ml_IN',
  'mn': 'mn_MN',
  'mr': 'mr_IN',
  'pl': 'pl_PL',
  'ps': 'ps_AF',
  'pt': 'pt_XX',
  'sv': 'sv_SE',
  'sw': 'sw_KE',
  'ta': 'ta_IN',
  'te': 'te_IN',
  'th': 'th_TH',
  'tl': 'tl_XX',
  'uk': 'uk_UA',
  'ur': 'ur_PK',
  'xh': 'xh_ZA',
  'gl': 'gl_ES',
  'sl': 'sl_SI'
}


def convert_to_pinyin(word):
    """
    Convert a Chinese word to its pinyin representation.
    
    Args:
        word (str): The Chinese word to convert.
        
    Returns:
        str: The pinyin representation of the word.
    """
    pinyin_list = pinyin(word)
    return ' '.join([syllable[0] for syllable in pinyin_list])


def download_spacy_model(model_name):
    """
    Download a spaCy model without displaying the download data.
    
    Args:
        model_name (str): The name of the spaCy model to download.
    """
    print("Downloading spaCy model...")
    # Suppress output by redirecting to os.devnull
    with open(os.devnull, 'w') as fnull:
        subprocess.run(
            ['python', '-m', 'spacy', 'download', model_name],
            stdout=fnull,
            stderr=fnull
        )


def get_language_full_name(short_code):
    """
    Get the full name of a language from its short code.
    
    Args:
        short_code (str): The short code of the language (e.g., 'zh', 'en').
        
    Returns:
        str: The full name of the language, or None if not found.
    """
    try:
        language = langcodes.Language.get(short_code)
        return language.display_name()
    except LookupError:
        return None


def load_spacy_model(model_name):
    """
    Load a spaCy model.
    
    Args:
        model_name (str): The name of the spaCy model to load.
        
    Returns:
        spacy.lang: The loaded spaCy model, or 1 if an error occurs.
    """
    try:
        print("Loading spaCy model...")
        nlp = spacy.load(model_name)
    except OSError:
        return 1
    return nlp


def remove_punctuation(text):
    """
    Remove punctuation from a given text.
    
    Args:
        text (str): The text from which to remove punctuation.
        
    Returns:
        str: The text without punctuation.
    """
    # Define a translation table to remove punctuation
    chinese_punctuation = "！？。，、；：“”‘’（）《》〈〉【】『』「」〔〕——…—"
    translator = str.maketrans('', '', string.punctuation + chinese_punctuation)
    return text.translate(translator)


def segment_text(text, lang):
    """
    Segment text into individual words and their parts of speech.
    
    Args:
        text (str): The text to segment.
        lang (str): The language of the text.
        
    Returns:
        list: A list of tuples containing segmented words and their parts of speech.
    """
    spacy_model_name = f'{lang}_core_web_sm'
    
    if lang == 'zh':
        words = pseg.lcut(text)
        return list(words)  # Keeping order by using list instead of set
    else:
        nlp = load_spacy_model(spacy_model_name)
        if nlp == 1:
            print(f"spaCy model '{spacy_model_name}' not found. Trying to download it.")
            download_spacy_model(spacy_model_name)
            nlp = load_spacy_model(spacy_model_name)
            if nlp == 1:
                print(f"spaCy model '{spacy_model_name}' not found. Trying to "
                      f"load another model: '{lang}_core_news_sm'")
                spacy_model_name = f'{lang}_core_news_sm'
                nlp = load_spacy_model(spacy_model_name)
                if nlp == 1:
                    print(f"spaCy model '{spacy_model_name}' not found. Trying to download it.")
                    download_spacy_model(spacy_model_name)
                    nlp = load_spacy_model(spacy_model_name)
                    if nlp == 1:
                        raise ValueError(f"Unsupported language: {lang}")
        print("")
        
        doc = nlp(text)
        return [(token.text, token.pos_) for token in doc]


def translate_text(text, translation_params):
    """
    Translate text using the specified translation parameters.
    
    Args:
        text (str): The text to translate.
        translation_params (dict): Parameters for translation, including:
            - src_lang (str): Source language.
            - target_lang (str): Target language.
            - translator (function): Translation function.
            - model (object): Translation model.
            - tokenizer (object): Tokenizer for the translation model.
            - chat (object): Chat object for translation.
            
    Returns:
        str: The translated text.
    """
    src_lang = translation_params["src_lang"]
    target_lang = translation_params["target_lang"]
    translator = translation_params["translator"]
    model = translation_params["model"]
    # facebook/mbart-large-50-many-to-many-mmt
    tokenizer = translation_params["tokenizer"]
    # Gemini 1.0 Pro
    chat = translation_params["chat"]
    # meta-llama/Meta-Llama-3-8B-Instruct
    pipeline = translation_params["pipeline"]
    terminators = translation_params["terminators"]
    
    if translator:
        translated = translator(text, max_length=512)
        return translated[0]['translation_text']
    elif model and tokenizer:
        encoded_text = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id[facebook_languages[target_lang]]
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    elif chat:
        response = chat.send_message(f"Translate this into {target_lang}: {text}")
        return response.text
    elif pipeline
    else:
        if model is None:
            raise ValueError("Missing model")
        else:
            raise ValueError("Missing tokenizer")


# Main function
def translate(src_text, src_lang, target_lang, model, add_pos=False, save_vocab=False):
    """
    Translate text from a source language to a target language using a specified model.

    Args:
        src_text (str): The source text to translate.
        src_lang (str): The source language code (e.g., 'en' for English).
        target_lang (str): The target language code (e.g., 'zh' for Chinese).
        model (str): The name of the translation model to use.
        add_pos (bool, optional): Whether to add parts of speech to the output. Defaults to False.
        save_vocab (bool, optional): Whether to save the vocabulary to a CSV file. Defaults to False.

    Raises:
        ValueError: If the source and target languages are the same or if an unsupported model is provided.

    Returns:
        None
    """
    if src_lang == target_lang:
        raise valueError("Source and target languages are the same!")
    
    # Initialize translation parameters
    translation_params = {
        'src_lang': src_lang,
        'target_lang': target_lang,
        'translator': None,
        'model': None,
        'tokenizer': None,
        'chat': None
    }

    src_lang_fullname = get_language_full_name(src_lang)
    target_lang_fullname = get_language_full_name(target_lang)

    if model.startswith("Helsinki-NLP/opus-mt"):
        # Initialize the translation pipeline
        translation_params["translator"] = pipeline("translation", model=model)
    elif model == "facebook/mbart-large-50-many-to-many-mmt":
        # Initialize mBART model and tokenizer
        translation_params["model"]  = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.src_lang = facebook_languages[src_lang]
        translation_params["tokenizer"] = tokenizer
    elif model == "Gemini 1.0 Pro":
        # Configure the Gemini model
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GEMINI_API_KEY")
        genai.configure(api_key = api_key)
        genmodel = genai.GenerativeModel(model_name='gemini-pro')
        chat = genmodel.start_chat()
        translation_params["chat"] = chat
    elif model == "meta-llama/Meta-Llama-3-8B-Instruct:
        # Create a text generation pipeline with the specified model
        # - Use bfloat16 precision for better performance and lower memory usage
        # - Automatically map the model to available GPUs
        #   References:
        #   1. https://huggingface.co/docs/accelerate/v0.25.0/en/concept_guides/big_model_inference
        #   2. https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling#using--accelerate
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        # Define terminators for the text generation
        # - eos_token_id: The token ID for the end-of-sequence token
        # - convert_tokens_to_ids("<|eot_id|>"): token to stop the inference
        #   See https://github.com/meta-llama/llama3/blob/main/README.md
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Translate the entire text to get context-aware translation
    full_translation = translate_text(src_text, translation_params)
    print(f"Source Text: {src_text}")
    print(f"Full Translation: {full_translation}\n")

    # Remove punctuation from the source text
    src_text = remove_punctuation(src_text)

    # Segment the source text into words with their parts of speech
    words = segment_text(src_text, src_lang)

    # Create a list to store the word details
    word_details = []

    # Print the vocabulary list with translations and pinyin (if it applies)
    for src_word, flag in words:
        if src_word in common_particles:
            translation = common_particles[src_word]
        else:
            translation = translate_text(src_word, translation_params)
            if src_lang == 'zh' and translation.lower() == "yes.":
                translation = "like"  # Fixing the incorrect translation for "喜欢"
        pos_full_name = POS_MAP.get(flag, 'unknown')
        word_detail = {
            f'{src_lang_fullname} Word': src_word,
        }
        word_detail_str = src_word
        if src_lang == 'zh':
            pinyin_src_word = convert_to_pinyin(src_word)
            word_detail.update({'Pinyin': pinyin_src_word})
            word_detail_str += f" ({pinyin_src_word})"
        word_detail.update({f'{target_lang_fullname} Translation': translation})
        word_detail_str += f": {translation} "
        if target_lang == 'zh':
            pinyin_translated_word = convert_to_pinyin(translation)
            word_detail.update({'Pinyin': pinyin_translated_word})
            word_detail_str += f"({pinyin_translated_word}) "
        if add_pos:
            word_detail.update({'Part of Speech': pos_full_name})
            word_detail_str += f"- {pos_full_name}"
        word_details.append(word_detail)
        print(word_detail_str)
        
    # Create a DataFrame from the word details
    if save_vocab:
        df = pd.DataFrame(word_details)

        # Save the DataFrame to a CSV file
        filename = f'{src_lang_fullname.lower()}_words_translation.csv'
        print(f"\nSaving csv file: {filename}")
        df.to_csv(f'{filename}', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    # Login to Hugging Face
    notebook_login()

    # Setup Pipeline and Tokenizer
    # To avoid: `RuntimeError: cutlassF: no kernel found to launch!`
    # Disable memory efficient and flash SDP (scaled dot product attention) to prevent runtime errors
    # See for more details:
    # - https://huggingface.co/stabilityai/stable-cascade/discussions/11
    # - https://pytorch.org/docs/stable/backends.html
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    # Test: Translate Chinese Text to English
    text = "你好！我叫王芳，我做了很多编程工作。"
    src_lang = "zh"
    target_lang = "en"

    model = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
    print(f"Model: {model}")
    translate(text, src_lang, target_lang, model=model)
