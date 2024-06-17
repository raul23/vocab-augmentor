import argparse
import csv
import importlib
import inspect
import logging
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import time
import warnings

from collections import defaultdict
from datetime import datetime

# Suppress the specific warning by setting the logging level to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

###################################
## Python Maps and Global Variables
###################################
AUDIO_MAIN_DIRPATH = "~/audio/"
GEN_MODEL = 'gemini'
LANG_DETECTOR = 'polyglot'
TRANSL_MODEL = 'helsinki'

LANG_DETECTORS = ['langdetect', 'langid', 'polyglot']

# ------------------------------------------------
# Mapping of short model names to full model names
# ------------------------------------------------
GEN_MODEL_MAP = {
    'gemini': 'gemini-pro',
    'llama': 'meta-llama/Meta-Llama-3-8B-Instruct'
}

TRANSL_MODEL_MAP = {
    'helsinki': 'Helsinki-NLP/opus-mt',
    'mbart': 'facebook/mbart-large-50-many-to-many-mmt',
    'gemini': 'gemini-pro',
    'llama': 'meta-llama/Meta-Llama-3-8B-Instruct'
}

# Table column information
COLUMN_TO_MIN_WIDTH = {
    'Word': 'min-width: 120px;', 
    'Pinyin (Word)': 'min-width: 120px;', 
    'Audio (Word)': 'min-width: 120px;',  
    'Translation': 'min-width: 120px;',
    'Pinyin (Translation)': 'min-width: 120px;', 
    'POS': 'min-width: 60px;', 
    'Word Origin': 'min-width: 120px;',
    'Example Sentences': 'min-width: 400px;',
    'Translation Model': 'min-width: 130px;', 
    'TTS Model': 'min-width: 100px;',
    'Sentence Generation Model': 'min-width: 190px;',
    'Date Added': 'min-width: 80px;',
    'Date Updated': 'min-width: 92px;',
    'Comments': 'min-width: 200px;',
}

# ------
# Colors
# ------
COLORS = {
    'GREEN': '\033[0;36m',  # 32
    'RED': '\033[0;31m',
    'YELLOW': '\033[0;33m',  # 32
    'BLUE': '\033[0;34m',  #
    'VIOLET': '\033[0;35m',  #
    'BOLD': '\033[1m',
    'NC': '\033[0m',
}
COLOR_TO_CODE = {
    'g': COLORS['GREEN'],
    'r': COLORS['RED'],
    'y': COLORS['YELLOW'],
    'b': COLORS['BLUE'],
    'v': COLORS['VIOLET'],
    'bold': COLORS['BOLD']
}

# Define a mapping for POS tags to their full names
# TODO: complete this list and test it
POS_MAP = {
    'a': 'adjective',
    'ad': 'adverbial',
    'ag': 'adjective morpheme',
    'an': 'nominal adjective',
    'aux': 'auxiliary',
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
COMMON_PARTICLES = {
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

FACEBOOK_LANGUAGES = {
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

##########################
# Classes: Language Models
##########################
class LanguageModel:
    def __init__(self, src_lang, target_lang, model_name):
        """
        Base class for language models that perform text generation and translation.

        Args:
            src_lang (str): Source language code (e.g., 'en', 'fr').
            target_lang (str): Target language code (e.g., 'zh', 'es').
            model_name (str): Name or identifier of the pretrained model.
        
        Attributes:
            src_lang_fullname (str): Full name of the source language.
            target_lang_fullname (str): Full name of the target language.
            model_name (str): Name or identifier of the pretrained model.
        """
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_fullname = get_language_full_name(src_lang)
        self.target_lang_fullname = get_language_full_name(target_lang)
        self.model_name = model_name
        
    def get_translation_request(self, text):
        request = f"Translate the following text into {self.target_lang_fullname} without any " + \
                  f"additional information or commentary: {text}"
        return request
        
    def get_text_generation_request(self, words, nb_sentences):
        """
        Generates a request template for text generation and translation.

        Args:
            words (str): Words to be included in the generated sentences.
            nb_sentences (int): Number of sentences to generate.
        
        Returns:
            str: The generated request template.
        """
        request = f"""
        Generate {nb_sentences} sentences in {self.src_lang_fullname} with the words '{words}' and
        provide their translations in {self.target_lang_fullname}.
        """
        if self.src_lang == 'zh' or  self.target_lang == 'zh':
            request += "If the sentence or translation is in Chinese, provide also its pinyin representation. "
        request += "Strictly provide only the sentences and their translations. " + \
                   "Do not include any introductions, explanations, examples, or notes."
        return request
    
    def generate_text(self, words, nb_sentences=2):
        """
        Generates text based on input words.

        Args:
            words (str): Words to be included in the generated sentences.
            nb_sentences (int, optional): Number of sentences to generate. Defaults to 2.
        
        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError("generate_text() not implemented!")
    
    def translate_text(self):
        """
        Translates text from source language to target language.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError("translate_text() not implemented!")
        

# facebook/mbart-large-50-many-to-many-mmt
class MBart(LanguageModel):
    def __init__(self, src_lang, target_lang):
        """
        Initializes the MBart class for multilingual translation.

        Args:
            src_lang (str): Source language code (e.g., 'en', 'fr').
            target_lang (str): Target language code (e.g., 'zh', 'es').
        """
        super().__init__(src_lang, target_lang, 
                         model_name="facebook/mbart-large-50-many-to-many-mmt")
        
        ## Import transformers and initialize MBart model and tokenizer
        # import transformers
        transformers = import_module("transformers")
        # from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        MBartForConditionalGeneration = import_module("transformers", "MBartForConditionalGeneration")
        MBart50TokenizerFast = import_module("transformers", "MBart50TokenizerFast")
        self.model  = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.tokenizer.src_lang = FACEBOOK_LANGUAGES[src_lang]
    
    def translate_text(self, text, return_tensors="pt", skip_special_tokens=True):
        """
        Translates text from source language to target language using MBart model.

        Args:
            text (str): Input text to be translated.
            return_tensors (str, optional): Return type of the generated tensor. Defaults to "pt".
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.
        
        Returns:
            str: Translated text in the target language.
        """
        encoded_text = self.tokenizer(text, return_tensors=return_tensors)
        generated_tokens = self.model.generate(
            **encoded_text,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[FACEBOOK_LANGUAGES[self.target_lang]]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=skip_special_tokens)[0]
    

# Helsinki-NLP/opus-mt
class HelsinkiNLP(LanguageModel):
    def __init__(self, src_lang, target_lang):
        """
        Initializes the HelsinkiNLP class for translation using Opus MT.

        Args:
            src_lang (str): Source language code (e.g., 'en', 'fr').
            target_lang (str): Target language code (e.g., 'zh', 'es').
        """
        # Check if the source language is an italic language and target language is english
        # See https://huggingface.co/Helsinki-NLP/opus-mt-itc-en
        if src_lang in ['it', 'ca', 'rm', 'es', 'ro', 'gl', 'sc', 'co', 'wa', 'pt', 'oc', 'an', 'id', 'fr', 'ht', 'itc', 'en'] \
            and target_lang == 'en':
            model_name = "Helsinki-NLP/opus-mt-itc-en"
        else:
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"
        
        super().__init__(src_lang, target_lang, model_name=model_name)

        ## Import transformers and initialize translation pipeline
        # import transformers
        transformers = import_module("transformers")
        # from transformers import pipeline
        pipeline = import_module("transformers", "pipeline")
        self.translator = pipeline("translation", model=self.model_name)
    
    def translate_text(self, text, max_length=512):
        """
        Translates text from source language to target language using Opus MT.

        Args:
            text (str): Input text to be translated.
            max_length (int, optional): Maximum length of the translated text. Defaults to 512.
        
        Returns:
            str: Translated text in the target language.
        """
        translated = self.translator(text, max_length=max_length)
        return translated[0]['translation_text']
    

# gemini-pro
class GeminiPro(LanguageModel):
    def __init__(self, src_lang, target_lang):
        """
        Initializes the GeminiPro class for text generation and translation.

        Args:
            src_lang (str): Source language code (e.g., 'en', 'fr').
            target_lang (str): Target language code (e.g., 'zh', 'es').
        """
        super().__init__(src_lang, target_lang, model_name="gemini-pro")
        
        ## Configure Gemini model using API key
        api_key = get_api_key("GEMINI_API_KEY")
        # import google.generativeai as genai
        genai = import_with_importlib("google.generativeai", alias="genai")
        genai.configure(api_key = api_key)
        del api_key
        
        # Initialize GenerativeModel for text generation
        genmodel = genai.GenerativeModel(model_name=self.model_name)
        self.chat = genmodel.start_chat()
    
    # TODO: add reference 
    def _send_message(self, request):
        """
        Sends a message to GeminiPro model for text generation or translation.

        Args:
            request (str): The request message for text generation or translation.
        
        Returns:
            str: Response message from GeminiPro model.
        """
        import google
        completed = False
        sleep = 0
        sleep_time = 2
        
        while not completed:
            try:
                response = self.chat.send_message(request)
            except google.api_core.exceptions.ResourceExhausted as re:
                print(f"ResourceExhausted exception occurred while processing property: {re}")
                sleep += 1
                if sleep > 5:
                    print(f"ResourceExhausted exception occurred 5 times in a row. Exiting.")
                    break
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                completed = True

        return response.text
    
    def generate_text(self, words, nb_sentences=2):
        """
        Generates text using GeminiPro model.

        Args:
            words (str): Words to be included in the generated sentences.
            nb_sentences (int, optional): Number of sentences to generate. Defaults to 2.
        
        Returns:
            str: Generated text.
        """
        request = self.get_text_generation_request(words, nb_sentences)
        return self._send_message(request)
    
    def translate_text(self, text):
        """
        Translates text using GeminiPro model.

        Args:
            text (str): Text to be translated.
        
        Returns:
            str: Translated text.
        """
        return self._send_message(self.get_translation_request(text))
    
    
# meta-llama/Meta-Llama-3-8B-Instruct
class Llama3(LanguageModel):
    def __init__(self, src_lang, target_lang):
        """
        Initializes the Llama3 class for text generation and translation.

        Args:
            src_lang (str): Source language code (e.g., 'en', 'fr').
            target_lang (str): Target language code (e.g., 'zh', 'es').
        """
        super().__init__(src_lang, target_lang, 
                         model_name="meta-llama/Meta-Llama-3-8B-Instruct")
        
        ## Import huggingface_hub and login with API key
        # from huggingface_hub import login
        login = import_module("huggingface_hub", "login")
        api_key = get_api_key("HF_API_KEY")
        login(api_key)
        del api_key
        
        # To avoid: `RuntimeError: cutlassF: no kernel found to launch!`
        # Disable memory efficient and flash SDP (scaled dot product attention) to prevent runtime errors
        # See for more details:
        # - https://huggingface.co/stabilityai/stable-cascade/discussions/11
        # - https://pytorch.org/docs/stable/backends.html
        # import torch
        torch = import_module("torch")
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        
        ## Setup text generation pipeline and tokenizer
        # Create a text generation pipeline with the specified model
        # - Use bfloat16 precision for better performance and lower memory usage
        # - Automatically map the model to available GPUs
        #   References:
        #   1. https://huggingface.co/docs/accelerate/v0.25.0/en/concept_guides/big_model_inference
        #   2. https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling#using--accelerate
        # import transformers
        transformers = import_module("transformers")
        #  To handle the warning "Setting pad_token_id to eos_token_id", explicitly specify the pad_token_id in the pipeline setup.
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            pad_token_id=transformers.AutoTokenizer.from_pretrained(self.model_name).eos_token_id
        )

        # Define terminators for the text generation
        # - eos_token_id: The token ID for the end-of-sequence token
        # - convert_tokens_to_ids("<|eot_id|>"): token to stop the inference
        #   See https://github.com/meta-llama/llama3/blob/main/README.md
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
    def _gen_text(self, request, context=None, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9):
        """
        Generates text using Llama3 model.

        Args:
            request (str): The request message for text generation.
            context (str, optional): Context information for text generation. Defaults to None.
            max_new_tokens (int, optional): Maximum number of new tokens. Defaults to 256.
            do_sample (bool, optional): Whether to sample outputs randomly. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.6.
            top_p (float, optional): Top-k sampling parameter. Defaults to 0.9.
        
        Returns:
            str: Generated text.
        """
        messages = [
            {"role": "user", "content": request},
        ]
        
        # If context is provided, add it to the messages
        if context:
            messages.append(
                {"role": "system", "content": context} # Add the context as a system message
            )
        
        # Generate text using the pipeline
        # See https://huggingface.co/docs/transformers/en/main_classes/text_generation
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        
        return outputs[0]["generated_text"][-1]["content"]
        
    def generate_text(self, words, nb_sentences=2, context=None, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9):
        """
        Generates text using Llama3 model.

        Args:
            words (str): Words to be included in the generated sentences.
            nb_sentences (int, optional): Number of sentences to generate. Defaults to 2.
            context (str, optional): Context information for text generation. Defaults to None.
            max_new_tokens (int, optional): Maximum number of new tokens. Defaults to 256.
            do_sample (bool, optional): Whether to sample outputs randomly. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.6.
            top_p (float, optional): Top-k sampling parameter. Defaults to 0.9.
        
        Returns:
            str: Generated text.
        """
        request = self.get_text_generation_request(words, nb_sentences)
        return self._gen_text(request, context, max_new_tokens, do_sample, temperature, top_p)
        
    
    def translate_text(self, text, context=None, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9):
        """
        Translates text using Llama3 model.

        Args:
            text (str): Text to be translated.
            context (str, optional): Context information for translation. Defaults to None.
            max_new_tokens (int, optional): Maximum number of new tokens. Defaults to 256.
            do_sample (bool, optional): Whether to sample outputs randomly. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.6.
            top_p (float, optional): Top-k sampling parameter. Defaults to 0.9.
        
        Returns:
            str: Translated text.
        """
        # Create the initial message with the user's request
        request = self.get_translation_request(text)
        return self._gen_text(request, context, max_new_tokens, do_sample, temperature, top_p)


#####################
# Classes: TTS models
#####################
class FacebookTTS:
    def __init__(self, model_name, verbose=False):
        """
        Initializes the FacebookTTS class with the specified model name.

        Args:
            model_name (str): The name of the pretrained model.
            verbose (bool, optional): Whether to print verbose outputs. Default is False.
        
        Attributes:
            torch: The imported torch module.
            scipy: The imported scipy module.
            np: The imported numpy module.
            model (VitsModel): The loaded VitsModel from transformers.
            tokenizer (AutoTokenizer): The loaded AutoTokenizer from transformers.
        """
        self.model_name = model_name
        
        ## Import necessary modules
        # from transformers import VitsModel, AutoTokenizer
        VitsModel = import_module("transformers", "VitsModel")
        AutoTokenizer = import_module("transformers", "AutoTokenizer")
        # import torch
        self.torch = import_module("torch")
        # import scipy
        self.scipy = import_module("scipy")
        # import numpy
        self.np = import_module("numpy", alias="np")
        
        # Load model and tokenizer
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_speech(self, text, **kwargs):
        """
        Generates speech from text input using the loaded model.

        Args:
            text (str): The input text to generate speech from.
            **kwargs: Additional keyword arguments for tokenization.
        
        Returns:
            str: The path to the generated audio file.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Generate waveform from model
        with self.torch.no_grad():
            output = self.model(**inputs).waveform
            
        # Generate a random temporary filename with .wav suffix
        audio_path = get_random_temp_filename(".wav")
        
        # Convert PyTorch tensor to NumPy array
        output_np = output.cpu().numpy()
        
        # Ensure the data is in the correct shape (1D array)
        if len(output_np.shape) > 1:
            output_np = output_np.squeeze()
        
        # Write the NumPy array to a WAV file
        self.scipy.io.wavfile.write(audio_path, rate=self.model.config.sampling_rate, data=output_np)
        
        return audio_path
       
    
class MeloTTS:
    def __init__(self, verbose=False):
        """
        Initializes the MeloTTS class with the specified verbosity setting.

        Args:
            verbose (bool, optional): Whether to print verbose outputs. Default is False.
        
        Attributes:
            client (Client): The Gradio Client instance for interacting with MeloTTS.
        """
        self.model_name = "MeloTTS"
        
        ## Import Gradio Client module
        # from gradio_client import Client
        Client = import_module("gradio_client", "Client")
        
        # Initialize Gradio Client for MeloTTS
        self.client = Client("mrfakename/MeloTTS", verbose=verbose)
    
    def generate_speech(self, text, lang, event_listener="predict"):
        """
        Generates speech from text input using the MeloTTS model.

        Args:
            text (str): The input text to generate speech from.
            lang (str): The language code for selecting the speaker.
            event_listener (str, optional): The type of event listener ('predict' or 'submit'). Default is 'predict'.
        
        Returns:
            str or Job: The path to the generated audio file if event_listener is 'predict',
                       or a job if event_listener is 'submit'.
        
        Raises:
            ValueError: If an unsupported event listener is provided.
        """
        # Determine speaker based on language
        if lang == "en":
            speaker = "EN-Default"
        else:
            speaker = lang.upper()
            
        # Perform prediction or submit job based on event_listener
        if event_listener == "predict":
            audio_path = self.client.predict(
                text=text,
                speaker=speaker,
                speed=1,
                language=lang.upper(),
                api_name="/synthesize"
            )
            return audio_path
        elif event_listener == "submit":
            job = self.client.submit(
                text=text,
                speaker=speaker,
                speed=1,
                language=lang.upper(),
                api_name="/synthesize"
            )
            return job
        else:
            raise ValueError(f"Unsupported event listener: {event_listener}")

    
###########
# Functions
###########
def color(msg, msg_color='y', bold_msg=False):
    """
    Applies color formatting to a message.

    Args:
        msg (str): The message to be colored.
        msg_color (str, optional): The color to apply to the message. Defaults to 'y'.
        bold_msg (bool, optional): Whether to make the message bold. Defaults to False.
    
    Returns:
        str: The colored message.
    
    Raises:
        AssertionError: If an unsupported color is specified.
    """
    msg_color = msg_color.lower()
    colors = list(COLOR_TO_CODE.keys())
    assert msg_color in colors, f'Wrong color: {msg_color}. Only these ' \
                                f'colors are supported: {colors}'
    msg = bold(msg) if bold_msg else msg
    msg = msg.replace(COLORS['NC'], COLORS['NC']+COLOR_TO_CODE[msg_color])
    return f"{COLOR_TO_CODE[msg_color]}{msg}{COLORS['NC']}"


def blue(msg):
    """
    Applies blue color to a message.

    Args:
        msg (str): The message to be colored in blue.
    
    Returns:
        str: The colored message in blue.
    """
    return color(msg, 'b')


def bold(msg):
    """
    Makes the message bold.

    Args:
        msg (str): The message to be made bold.
    
    Returns:
        str: The bold message.
    """
    return color(msg, 'bold')


def green(msg):
    """
    Applies green color to a message.

    Args:
        msg (str): The message to be colored in green.
    
    Returns:
        str: The colored message in green.
    """
    return color(msg, 'g')


def red(msg):
    """
    Applies red color to a message.

    Args:
        msg (str): The message to be colored in red.
    
    Returns:
        str: The colored message in red.
    """
    return color(msg, 'r')


def violet(msg):
    """
    Applies violet color to a message.

    Args:
        msg (str): The message to be colored in violet.
    
    Returns:
        str: The colored message in violet.
    """
    return color(msg, 'v')


def yellow(msg):
    """
    Applies yellow color to a message.

    Args:
        msg (str): The message to be colored in yellow.
    
    Returns:
        str: The colored message in yellow.
    """
    return color(msg)


def build_index(file_path):
    """
    Builds an index from the CSV file where each key is a word from the "Word" column,
    and each value is a list of row numbers that contain that word.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        dict: An index with words as keys and lists of row numbers as values.
        list: All rows in the CSV file.
    
    Raises:
        IOError: If an error occurs while reading the file.
    """
    if not os.path.exists(file_path):
        # raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        return {}, []
    
    index = defaultdict(list)
    rows = []
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                rows.append(row)
                index[row['Word']].append(i)
    except IOError as e:
        raise IOError(f"An error occurred while reading the file: {e}")
    
    return index, rows


# TODO: not used
def create_temp_file(data, mode='w+b'):
    """
    Creates a temporary file with the given data in a temporary folder with a random filename.
    
    Args:
        data (str or bytes): The data to be written to the temporary file.
        mode (str): The mode in which the file is opened. Default is 'w+b' for binary writing.
    
    Returns:
        str: The path to the created temporary file, or None if an error occurs.
        
    Raises:
        Exception: If an error occurs during file creation.
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode=mode) as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name
        #print(f"Temporary file created successfully at: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def create_text_file(content, path):
    """
    Creates a text file with the given content at the specified path.
    
    Args:
        content (str): The content to be written to the text file.
        path (str): The path where the text file will be saved.
    
    Returns:
        str: The path to the created text file, or None if an error occurs.
        
    Raises:
        Exception: If an error occurs during file creation.
    """
    try:
        with open(path, 'w') as file:
            file.write(content)
        # print(f"File created successfully at: {path}")
        return path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def detect_language(text, lang_detector):
    """
    Detects the language of the given text using the specified language detection method.

    Args:
        text (str): The text whose language needs to be detected.
        lang_detector (str): The language detection method to use. Supported values are 'langdetect', 'langid', or 'polyglot'.

    Returns:
        str or None: The detected language code (e.g., 'en' for English), or None if language detection fails.

    Raises:
        ValueError: If an unsupported language detector is provided.
    """
    if lang_detector == "langdetect":
        return detect_language_with_langdetect(text)
    elif lang_detector == "langid":
        return detect_language_with_langid(text)
    elif lang_detector == "polyglot":
        return detect_language_with_polyglot(text)
    else:
        raise ValueError(f"Unsupported language detector: {lang_detector}")
    

def detect_language_with_langdetect(text):
    """
    Detects the language of the given text using the langdetect library.

    Args:
        text (str): The text whose language needs to be detected.

    Returns:
        str or None: The detected language code (e.g., 'en' for English), or None if language detection fails.
        
    Raises:
        LangDetectException: If language detection fails.
    """
    # langdetect fails with `text = "今天我很高兴"`, it detects it as ko (korean)
    # detects `text = "Estou feliz"` as pt

    # Install langdetect
    verify_and_install_packages(["langdetect"])
    
    ## Imports
    # from langdetect import detect, DetectorFactory
    detect = import_module("langdetect", "detect")
    DetectorFactory = import_module("langdetect", "DetectorFactory")
    # from langdetect import lang_detect_exception
    lang_detect_exception = import_module("langdetect", "lang_detect_exception")

    # For langdetect
    DetectorFactory.seed = 0

    try:
        # Detect the language
        # Will return zh-cn and zh-tw (not a 2-letters code), see https://pypi.org/project/langdetect/
        language_code = detect(text)
        return language_code
    except lang_detect_exception.LangDetectException as e:
        # Handle cases where the language could not be detected
        print(f"Language detection failed: {e}")
        return None


def detect_language_with_langid(text):
    """
    Detects the language of the given text using the langid library.

    Args:
        text (str): The text whose language needs to be detected.

    Returns:
        str or None: The detected language code (e.g., 'en' for English), or None if language detection fails.
        
    Raises:
        error: If language detection fails.
    """
    # langid detects `text = "今天我很高兴"` as zh (chinese)
    # fails to detet `text = "Estou feliz"` as pt, detects it as cy
    
    # Install languid
    verify_and_install_packages(["langid"])
    
    # import langid
    langid = import_module("langid")

    try:
        # Detect the language
        language_code, _ = langid.classify(text)
        return language_code
    except error as e:
        # TODO: don't use generic exception error
        # Handle case where the language could not be detected
        print(f"Language detection failed: {e}")
        return None


def detect_language_with_polyglot(text):
    """
    Detects the language of the given text using the polyglot library.

    Args:
        text (str): The text whose language needs to be detected.

    Returns:
        str or None: The detected language code (e.g., 'en' for English), or None if language detection fails.
        
    Raises:
        error: If language detection fails.
    """
    # polyglot detects `text = "今天我很高兴"` as chinese and `text = "Estou feliz"` as pt
    
    # Install polyglot and its necessary packages
    # TODO important: are these necessary packages?
    verify_and_install_packages(["polyglot", "pyicu", "pycld2", "pycld3"])
    
    ## Imports
    # from polyglot.detect import Detector
    Detector = import_module("polyglot.detect", "Detector")
    # from polyglot.detect.base import logger
    logger = import_module("polyglot.detect.base", "logger")

    # Suppress polyglot logging
    logger.setLevel(logging.ERROR)    
    
    try:
        # Detect the language
        detector = Detector(text)
        language_code = detector.language.code
        return language_code
    except error as e:
        # TODO: don't use generic exception error
        # Handle case where the language could not be detected
        print(f"Language detection failed: {e}")
        return None


def download_spacy_model(model_name):
    """
    Download a spaCy model without displaying the download data.
    
    Args:
        model_name (str): The name of the spaCy model to download.
    """
    # print("Downloading spaCy model...")
    # Suppress output by redirecting to os.devnull
    with open(os.devnull, 'w') as fnull:
        subprocess.run(
            ['python', '-m', 'spacy', 'download', model_name],
            stdout=fnull,
            stderr=fnull
        )


def get_api_key(token_name):
    """
    Retrieves an API key from environment variables or Kaggle Secrets.

    Args:
        token_name (str): The name of the token or API key to retrieve.

    Returns:
        str or None: The API key retrieved from environment variables or Kaggle Secrets,
                     or None if the key could not be found.
    """
    api_key = os.environ.get(token_name)
    if api_key is None:
        print(f"Couldn't find {token_name} in environment variables")
        print(f"Reading {token_name} from Kaggle Secrets...")
        # from kaggle_secrets import UserSecretsClient
        UserSecretsClient = import_module("kaggle_secrets", "UserSecretsClient")
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret(token_name)
    return api_key
        

def get_filepath_without_extension(filepath):
    """
    Get the file path without the file extension.

    Args:
        filepath (str): The full file path.

    Returns:
        str: The file path without the extension.
    """
    root, _ = os.path.splitext(filepath)
    return root


def get_language_full_name(short_code):
    """
    Get the full name of a language from its short code.
    
    Args:
        short_code (str): The short code of the language (e.g., 'zh', 'en').
        
    Returns:
        str: The full name of the language, or None if not found.
        
    Raises:
        LookupError: If the language short code is not recognized.
    """
    langcodes = import_module("langcodes")
    
    try:
        language = langcodes.Language.get(short_code)
        return language.display_name()
    except LookupError:
        return None

    
def get_random_temp_filename(suffix=""):
    """
    Generates a random temporary filename without creating the file.
    
    Returns:
        str: The generated temporary filename.
        
    Raises:
        Exception: If an error occurs during filename generation.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            temp_file_path = temp_file.name
        #print(f"Generated temporary filename: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def import_module(module_name, attribute_name=None, alias=None, install_module=True):
    """
    Imports a Python module dynamically, optionally importing specific attributes or using an alias.
    If the module is not found, it will be installed if requested.

    Args:
        module_name (str): The name of the module to import.
        attribute_name (str, optional): The name of the specific attribute to import from the module.
        alias (str, optional): An alias to assign to the imported module or attribute.
        install (bool, optional): Whether to install the module if it couldn't be imported.

    Returns:
        module or object: The imported module or attribute, or None if the module couldn't be imported.
        
    Raises:
        ImportError: If the module couldn't be imported.
    """
    try:
        module = import_with_importlib(module_name, attribute_name, alias)
    except ImportError as e:
        if install_module:
            print(f"{module_name} module not found. Installing...")
            install(module_name)
            module = import_with_importlib(module_name, attribute_name, alias)
        else:
            # TODO: test raising this error
            raise ImportError(e)
    return module
    

def import_with_importlib(module_name, attribute_name=None, alias=None):
    """
    Dynamically import a module or an attribute from a module using importlib.
    
    Args:
        module_name (str): The name of the module to import.
        attribute_name (str, optional): The attribute of the module to import. Defaults to None.
        alias (str, optional): The alias to assign to the imported module or attribute. Defaults to None.
    
    Returns:
        module or attribute: The imported module or attribute.
    """
    module = importlib.import_module(module_name)
    if attribute_name:
        item = getattr(module, attribute_name)
        if alias:
            globals()[alias] = item
        return item
    if alias:
        globals()[alias] = module
    return module


def install(package):
    """
    Installs a Python package using pip.

    Args:
        package (str): The name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("\n")


def is_package_installed(package_name):
    """
    Check if a package is installed without actualy importing it, 
    which is more memory-efficient.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    # Known aliases for certain packages
    package_aliases = {
        'pycld3': 'cld3',
        'pyicu': 'icu',
    }
    
    # Use alias if available
    if package_name in package_aliases:
        package_name = package_aliases[package_name]
    
    # First, try to find the spec of the package
    package_spec = importlib.util.find_spec(package_name)
    if package_spec is not None:
        return True

    # If spec is None, try to import the package
    try:
        __import__(package_name)
        return True
    except ModuleNotFoundError:
        return False


def move_file(source_path, destination_path):
    """
    Moves a file from source_path to destination_path.
    
    Args:
        source_path (str): The path to the source file.
        destination_filename (str): The path to the file at the destination.
    
    Returns:
        str: The path to the moved file.
    """
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Move the file
    shutil.move(source_path, destination_path)
    
    return destination_path


# TODO: not used
def read_csv(filename):
    """
    Reads data from a CSV file and returns it as a list of dictionaries.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries representing rows from the CSV file.
              An empty list is returned if the file does not exist.
    """
    if not os.path.exists(filename):
        return []
    
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data
    
    
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


def rename_file(file_path, new_file_name):
    """
    Renames a file given its current path and new file name.

    Args:
        file_path (str): The current path of the file.
        new_file_name (str): The new name to assign to the file.

    Returns:
        str: The new file path after renaming.
    """
    # Get the directory
    directory, _ = os.path.split(file_path)
    
    # Create the full path for the new file
    new_file_path = os.path.join(directory, new_file_name)
    
    # Rename the file
    os.rename(file_path, new_file_path)
    
    # print(f'Renamed file: {file_path} -> {new_file_path}')
    return new_file_path


# TODO: not used
def save_updated_csv(file_path, rows):
    """
    Saves the updated rows back to the CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        rows (list): The list of all rows to be saved.
        
    Raises:
        IOError: If an error occurs while writing to the CSV file.
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except IOError as e:
        raise IOError(f"An error occurred while writing to the file: {e}")
        
        
def search_csv_rows(index, search_word):
    """
    Searches for a word in the index.
    
    Args:
        index (dict): The index built from the CSV file.
        search_word (str): The word to search for.
    
    Returns:
        list:
    """
    # Return row indices where the word was found
    return index.get(search_word, [])


def verify_and_install_packages(packages):
    for pkg in packages:
        if not is_package_installed(pkg):
            install(pkg)
            
            
# TODO: not used
def write_csv(filename, data, fieldnames):
    """
    Writes data to a CSV file.

    Args:
        filename (str): The path to the CSV file.
        data (list of dict): The data to write, where each dictionary represents a row.
        fieldnames (list of str): The field names to use in the CSV header.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


###########################################
# Functions and Classes for Argument Parser
###########################################
def get_default_message(default_value):
    """
    Generate a formatted default message.

    This function returns a default message formatted with the given default value
    in green color.

    Args:
        default_value (str): The default value to be included in the message.

    Returns:
        str: The formatted default message.
    """
    return green(f' (default: {default_value})')


class ArgumentParser(argparse.ArgumentParser):
    """
    A custom ArgumentParser to provide a custom error message format.

    This class overrides the default error method to print the usage message
    and a custom error message in red color when an error occurs.
    """

    def error(self, message):
        """
        Handle an error in argument parsing.

        This method prints the usage message and a custom error message in red color,
        then exits the program with status code 2.

        Args:
            message (str): The error message to be displayed.
        """
        print(self.format_usage().splitlines()[0])
        self.exit(2, red(f'\nerror: {message}\n'))


class MyFormatter(argparse.HelpFormatter):
    """
    A custom HelpFormatter to correct the maximum action length for the indenting of subactions.

    This formatter adjusts the maximum action length to account for the additional indentation
    of subactions, ensuring that help messages are properly aligned.
    """

    def add_argument(self, action):
        """
        Add an argument to the parser, adjusting the maximum action length for subactions.

        This method overrides the default add_argument method to account for the additional
        indentation of subactions, ensuring that the help messages are properly aligned.

        Args:
            action (argparse.Action): The argument action to be added.
        """
        if action.help is not argparse.SUPPRESS:

            # find all invocations
            get_invocation = self._format_action_invocation
            invocations = [get_invocation(action)]
            current_indent = self._current_indent
            for subaction in self._iter_indented_subactions(action):
                # compensate for the indent that will be added
                indent_chg = self._current_indent - current_indent
                added_indent = 'x' * indent_chg
                invocations.append(added_indent + get_invocation(subaction))
            # print('inv', invocations)

            # update the maximum item length
            invocation_length = max([len(s) for s in invocations])
            action_length = invocation_length + self._current_indent
            self._action_max_length = max(self._action_max_length,
                                          action_length)

            # add the item to the list
            self._add_item(self._format_action, [action])

    # Ref.: https://stackoverflow.com/a/23941599/14664104
    def _format_action_invocation(self, action):
        """
        Format the action invocation string.

        This method formats the invocation string for an action, taking into account whether
        the action has option strings and whether it takes a value.

        Args:
            action (argparse.Action): The argument action to be formatted.

        Returns:
            str: The formatted action invocation string.
        """
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # parts.append('%s %s' % (option_string, args_string))
                    parts.append('%s' % option_string)
                parts[-1] += ' %s'%args_string
            return ', '.join(parts)

        
# Ref.: https://stackoverflow.com/a/4195302/14664104
def required_length(nmin, nmax, is_list=True):
    """
    Define a custom argparse action to enforce the number of arguments.

    This function creates and returns a custom argparse action that enforces
    the requirement that a specific number of arguments are provided. If the
    number of arguments is not within the specified range, an error is raised.

    Args:
        nmin (int): The minimum number of arguments required.
        nmax (int): The maximum number of arguments allowed.
        is_list (bool): A flag indicating whether the input should be treated as a list.

    Returns:
        argparse.Action: A custom argparse action enforcing the specified argument length.
    """
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if isinstance(values, str):
                tmp_values = [values]
            else:
                tmp_values = values
            if not nmin <= len(tmp_values) <= nmax:
                if nmin == nmax:
                    msg = 'argument "{f}" requires {nmin} arguments'.format(
                        f=self.dest, nmin=nmin, nmax=nmax)
                else:
                    msg = 'argument "{f}" requires between {nmin} and {nmax} ' \
                          'arguments'.format(f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength
        
        
def setup_argparser():
    """
    Set up the argument parser for the vocabulary expansion tool.

    This function configures the argument parser to handle command-line arguments
    for processing, translating, and managing vocabulary from the provided text.

    Returns:
        ArgumentParser: Configured argument parser with the required options.
    """
    try:
        width = os.get_terminal_size().columns - 5
        usage_msg = '%(prog)s '
    except OSError:
        # No access to terminal
        width = 110
        usage_msg = 'vocab '
    usage_msg += '[OPTIONS] {TEXT} {TGT_LANG}'
    desc_msg = 'Expand your vocabulary list by identifying and translating new words from provided text using various language models.'
    parser = ArgumentParser(
        description="",
        usage=blue(f"{usage_msg}\n\n{desc_msg}"),
        add_help=False,
        formatter_class=lambda prog: MyFormatter(
            prog, max_help_position=50, width=width))
    parser.add_argument(
        '-h', '--help', action='help',
        help='Display detailed usage instructions and exit the program.')
    parser.add_argument(
        '-t', '--text', metavar='TEXT', type=str, required=True, 
        help='The source text that will be processed to identify and translate new words.')
    parser.add_argument(
        '-l', '--target_lang', metavar='TGT_LANG', type=str, required=True, 
        help='''Target language code into which the source text will be translated (e.g., zh for Chinese, en for English,
             pt for Portuguese).''')
    parser.add_argument(
        '-o', '--text_origin', metavar='ORIGIN', type=str, 
        help='Origin of the source text, e.g. movie script, book, URL of website, etc.')
    parser.add_argument(
        '-d', '--lang_detector', metavar='NAME', type=str, default=LANG_DETECTOR, 
        choices=LANG_DETECTORS,
        help='Method to use for detecting the language of the source text.' 
             + get_default_message(LANG_DETECTOR))
    parser.add_argument(
        '-m', '--transl_model', metavar='NAME', type=str, default=TRANSL_MODEL, 
        choices=list(TRANSL_MODEL_MAP.keys()),
        help='Translation model to use for translating the text.'
             + get_default_message(TRANSL_MODEL))
    parser.add_argument(
        '-g', '--gen_model', metavar='NAME', type=str, default=GEN_MODEL,
        choices=list(GEN_MODEL_MAP.keys()),
        help='Language model to use for generating example sentences in the source language.'
             + get_default_message(GEN_MODEL))
    parser.add_argument(
        '-c', '--csv_filepath', metavar='CSV_FILE', type=str,
        help='Path to the vocabulary CSV file. If the file does not exist, a new one will be created.')
    parser.add_argument(
        '-a', '--audio_dirpath', metavar='AUDIO_DIR', type=str,
        help='Path to the main directory for storing audio files.' + get_default_message(AUDIO_MAIN_DIRPATH))
    parser.add_argument(
        '-b', '--audio_base_url', metavar='URL', type=str,
        help='Base URL to audio files of words. (experimental)')
    parser.add_argument(
        '--ap', '--add_pos', dest='add_pos', action='store_true', 
        help='Flag to add or update part-of-speech (POS) information for the words.')
    parser.add_argument(
        '--as', '--add_sentences',  dest='add_sentences', action='store_true', 
        help='Flag to add or update example sentences in the vocabulary list.')
    parser.add_argument(
        '--aut', '--add_audio_text', dest='add_audio_text', action='store_true', 
        help='Flag to add or update audio pronunciation for the source text.')
    parser.add_argument(
        '--aaw', '--add_audio_words', dest='add_audio_words', action='store_true', 
        help='Flag to add or update audio pronunciation for the extracted words from the text.')     
    parser.add_argument(
        '--ascb', '--add_save_comments_button', dest='add_save_comments_button', action='store_true', 
        help="Flag to add 'Save Comments' button in the HTML page of the table. (experimental)") 
    return parser


############
# Main Class
############
class VocabAugmentor:
    """
    A class used to augment vocabulary with various functionalities including
    translation, text-to-speech, and text segmentation.

    Attributes
    ----------
    pd : module
        Pandas module for data manipulation.
    pinyin : module
        Module for converting Chinese characters to pinyin.
    pseg : module
        Module for segmentation of Chinese text.
    spacy : module
        Module for segmentation of non-Chinese text.
    transl_model_name : str
        Name of the translation model.
    transl_model : object
        Translation model object.
    gen_model_name : str
        Name of the generation model.
    gen_model : object
        Generation model object.
    tts_model_name : str
        Name of the text-to-speech model.
    tts_model : object
        Text-to-speech model object.
    lang_detector : str
        Name of language detector method.
    audio_main_dirpath : str
        Path to the main directory for storing audio files.
    audio_text_dirpath : str
        Directory for storing audio of text.
    audio_text_filepath : str
        File path for the audio of text.
    audio_text_content_filepath : str
        File path for the content of the text to be converted to audio.
    audio_text_content : str
        Content of the text to be converted to audio.
    audio_base_url : str
        Base URL for links to audio files of words. (experimental)
    audio_words_dirpath : str
        Directory for storing audio of words.
    vocab_csv_filepath : str
        Path to the vocabulary CSV file.
    add_pos : bool
        Flag indicating whether to add part-of-speech tags.
    add_sentences : bool
        Flag indicating whether to add sentences.
    add_audio_text : bool
        Flag indicating whether to add audio for the text.
    add_audio_words : bool
        Flag indicating whether to add audio for individual words.
    add_save_comments_button: bool
        Flag indicating whether to add the 'Save Comments' button in the HTML page of the table. (experimental)
    src_text : str
        Source text.
    src_text_no_punct : str
        Source text without punctuation.
    src_text_pinyin : str
        Pinyin representation of the source text.
    src_text_origin : str
        Origin of the source text.
    full_translation : str
        Full translation of the source text.
    full_translation_pinyin : str
        Pinyin representation of the full translation.
    src_lang : str
        Source language code.
    src_lang_fullname : str
        Full name of the source language.
    target_lang : str
        Target language code.
    target_lang_fullname : str
        Full name of the target language.
    audio_jobs : list
        List of audio jobs.
    chinese_chars_to_pinyin : dict
        Dictionary mapping Chinese characters to pinyin.
    """
    
    def __init__(self):
        """
        Initializes the VocabAugmentor object with default values for its attributes.
        """
        ## Modules
        self.pd = None  # pandas
        self.pinyin = None
        self.pseg = None  # For segmentation of Chinese text
        self.spacy = None  # For segmentation of non-Chinese text
        
        ## Models
        self.transl_model_name = ""
        self.transl_model = None
        self.gen_model_name = ""
        self.gen_model = None
        self.tts_model_name = ""
        self.tts_model = None
        self.lang_detector = ""
        
        ## Directories and files
        self.audio_main_dirpath = ""
        self.audio_text_dirpath = ""
        self.audio_text_filepath = ""
        self.audio_text_content = ""
        self.audio_text_content_filepath = ""
        self.audio_words_dirpath = ""
        self.vocab_csv_filepath = ""
        self.audio_base_url = ""  # (experimental)
        
        ## Flags
        self.add_pos = False
        self.add_sentences = False
        self.add_audio_text = False
        self.add_audio_words = False
        self.add_save_comments_button = False  # (experimental)
        
        ## Source text and its translation
        self.src_text = ""
        self.src_text_no_punct = ""
        self.src_text_pinyin = ""
        self.src_text_origin = ""
        self.full_translation = ""
        self.full_translation_pinyin = ""
        self.src_lang = ""
        self.src_lang_fullname = ""
        self.target_lang = ""
        self.target_lang_fullname = ""
        
        # List and dict
        self.audio_jobs = []
        self.chinese_chars_to_pinyin ={}
    
    def _convert_to_pinyin(self, word):
        """
        Convert a Chinese word to its pinyin representation.

        Args:
            word (str): The Chinese word to convert.

        Returns:
            str: The pinyin representation of the word.
        """
        # Import/install pypinyin module
        if self.pinyin is None:
            # from pypinyin import pinyin
            self.pinyin = import_module("pypinyin", "pinyin")
        pinyin_word = self.chinese_chars_to_pinyin.get(word)
        if pinyin_word is None: 
            # Compute Pinyin for the word
            pinyin_list = self.pinyin(word)
            pinyin_word = ' '.join([syllable[0] for syllable in pinyin_list])
            # Save the computed Pinyin for later lookup
            self.chinese_chars_to_pinyin.setdefault(word, pinyin_word)    
        return pinyin_word
    
    @staticmethod
    def _is_language_supported_by_tts(lang):
        """
        Check if a language is supported by the text-to-speech (TTS) models.

        Args:
            lang (str): Language code to check.

        Returns:
            bool: True if the language is supported, False otherwise.
        """
        melotts_lang = ['EN', 'ES', 'FR', 'ZH', 'JP', 'KR']
        facebooktts_lang = ['PT']
        all_supported_languages = melotts_lang + facebooktts_lang
        if lang.upper() in all_supported_languages:
            return True
        else:
            print(f"The language '{lang}' is not supported by none of the TTS")
            print(f"These are the supported languages: {all_supported_languages}")
            return False

    def _load_spacy_model(self, model_name):
        """
        Load a spaCy model.

        Args:
            model_name (str): The name of the spaCy model to load.

        Returns:
            spacy.lang: The loaded spaCy model, or 1 if an error occurs.
        """
        try:
            nlp = self.spacy.load(model_name)
        except OSError:
            return 1
        return nlp
    
    def _load_gen_model(self):
        """
        Load the sentence generation model.
        
        Returns:
            None
        """
        # Check if there is already a loaded sentence generation model that matches the chosen configuration
        if self.gen_model and self.gen_model.model_name == self.gen_model_name and \
            self.gen_model.src_lang == self.src_lang and self.gen_model.target_lang == self.target_lang:
            pass
        elif self.add_sentences:
            if self.gen_model_name == "gemini-pro":
                if self.transl_model_name == "gemini-pro":
                    self.gen_model = self.transl_model
                else:
                    self.gen_model = GeminiPro(self.src_lang, self.target_lang)
            elif self.gen_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                if self.transl_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                    self.gen_model = self.transl_model
                else:
                    self.gen_model = Llama3(self.src_lang, self.target_lang)
            else:
                raise ValueError(f"Unsupported sentence generation model: {self.gen_model_name}")
        else:
            self.gen_model = None
            self.gen_model_name = ""
            
    def _load_transl_model(self):
        """
        Load the translation model.
        
        Returns:
            None
        """
        # Check if there is already a loaded translation model that matches the chosen configuration
        if self.transl_model and self.transl_model.model_name == self.transl_model_name and \
            self.transl_model.src_lang == self.src_lang and self.transl_model.target_lang == self.target_lang:
            pass
        elif self.transl_model_name == "Helsinki-NLP/opus-mt":
            self.transl_model = HelsinkiNLP(self.src_lang, self.target_lang)
        elif self.transl_model_name == "facebook/mbart-large-50-many-to-many-mmt":
            self.transl_model = MBart(self.src_lang, self.target_lang)
        elif self.transl_model_name == "gemini-pro":
            self.transl_model = GeminiPro(self.src_lang, self.target_lang)
        elif self.transl_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.transl_model = Llama3(self.src_lang, self.target_lang)
        else:
            raise ValueError(f"Unsupported translatation model: {self.transl_model_name}")
    
    def _load_TTS_model(self):
        """
        Load the text-to-speech (TTS) model and create audio directories if needed.

        Sets flags to disable audio generation if the source language is not supported.
        
        Returns:
            None
        """
        if self.add_audio_text or self.add_audio_words:
            # Check if the source language is supported by one of the TTS model
            if self._is_language_supported_by_tts(self.src_lang):
                ## Load TTS model
                if self.src_lang == "pt":
                    # Check first if there is already a loaded TTS model
                    if self.tts_model and self.tts_model.model_name == "facebook/mms-tts-por":
                        pass
                    else:
                        # Portuguese supported by Facebook-TTS (not MeloTTS)
                        self.tts_model = FacebookTTS("facebook/mms-tts-por")
                else:
                    # Check first if there is already a loaded TTS model
                    if self.tts_model and self.tts_model.model_name == "MeloTTS":
                        pass
                    else:
                        self.tts_model = MeloTTS()
                self.tts_model_name = self.tts_model.model_name
            else:
                # Source language not supported, thus audio generation disabled completely
                self.add_audio_words = False
                self.add_audio_text = False
                
            # Creation of audio directories
            if self.add_audio_text:
                # Create the audio directory for text sounds
                self.audio_text_dirpath = os.path.join(self.audio_main_dirpath, f"{self.src_lang_fullname.lower()}/text")
                os.makedirs(self.audio_text_dirpath, exist_ok=True)
            if self.add_audio_words:
                # Create the audio directory for word sounds
                self.audio_words_dirpath = os.path.join(self.audio_main_dirpath, f"{self.src_lang_fullname.lower()}/words")
                os.makedirs(self.audio_words_dirpath, exist_ok=True)
    
    def _print_entry(self, entry):
        """
        Print the information of a given vocabulary entry.

        Args:
            entry (dict): A dictionary containing the details of a vocabulary entry.
            
        Returns:
            None
        """
        # String containing all info (e.g. Pinyin, POS, Dates, ...) for a given source word to be printed
        print_str = ""
        
        # Print first the source word and its Pinyin (if applicable)
        print_str += blue(f"{entry['Word']}")
        if self.src_lang == 'zh':
            print_str += f" ({entry['Pinyin (Word)']}): "
        else:
            print_str += f": "
            
        # Print the translated word and its Pinyin(if applicable)
        print_str += f"{entry['Translation']}"
        if self.target_lang == 'zh':
            print_str += f" ({entry['Pinyin (Translation)']})"
        
        if self.add_pos:
            print_str += f" - {entry['POS']}"
            
        if self.add_sentences:
            print_str += f"\nExample Sentences:\n{entry['Example Sentences']}\n\n"
            
        print(print_str)
        
    
    def _process_audio_text(self):
        """
        Generate and process the audio for the source text if needed.
        
        Returns:
            None
        """
        # Generate speech for the source text if:
        # 1. The corresponding flag is enabled
        #               AND
        # 2. The audio path doesn't exist
        if self.add_audio_text:
            if not os.path.exists(self.audio_text_filepath):
                # Use source text with punctuations when generating speech
                source_path = self.tts_model.generate_speech(
                    text=self.src_text, lang=self.src_lang, event_listener="predict")
                move_file(source_path, self.audio_text_filepath)
                # TODO: log following info instead
                print(f"Audio file (text): {self.audio_text_filepath}")
            if not os.path.exists(self.audio_text_content_filepath):
                create_text_file(self.audio_text_content, self.audio_text_content_filepath)
                print(f"Text file: {self.audio_text_content_filepath}")
    
    def _process_audio_words(self):
        """
        Generate and process the audio for individual words if needed.

        Moves the generated audio files to the specified directory.
        
        Returns:
            None
        """
        if self.add_audio_words:
            for audio_path, job in self.audio_jobs:
                # source_path is a temporary path for the audio file
                # audio_path is the destination path for the audio file
                if type(job) == str:
                    source_path = job
                else:
                    source_path = job.result()
                #ipdb.set_trace()
                if not os.path.exists(audio_path):
                    move_file(source_path, audio_path)
                    # TODO: log following info instead
                    print(f"Audio file (word): {audio_path}")
                
    def _process_extracted_words(self, words):
        """
        Process extracted words, updating or adding them to the vocabulary list.

        Args:
            words (list of tuple): A list of tuples where each tuple contains a source word and its POS tag.
            
        Returns:
            csv_rows: TODO
        """
        index = 0
        new_entries = []
        new_words = []
        self.audio_jobs = []
        self.chinese_chars_to_pinyin ={}
        
        init_entry_values = {}
        for k,v in COLUMN_TO_MIN_WIDTH.items():
            if k == "Date Added":
                init_entry_values[k] = datetime.now().strftime("%Y-%m-%d")
            else:
                init_entry_values[k] = ''
        
        # Build an in-memory index for faster subsequent searches where:
        # - each key is a word from the "Word" column
        # - each value is a list of rows that contain that word
        # csv_rows contain all rows in the CSV file
        csv_index, csv_rows = build_index(self.vocab_csv_filepath)
        
        # Each extracted word is associated with a POS tag (e.g. a, ad)
        for i, (src_word, pos_tag) in enumerate(words):
            # Check if src_word is a new word but was already processed
            if src_word in new_words:
                continue
                
            pos_tag = pos_tag.lower()
            
            # Search all the CSV row indices for rows that have the `src_word` in the 'Word' column
            # NOTE: src_word can appear more than once in the csv file, e.g. the user added them manually
            row_indices = search_csv_rows(csv_index, src_word)
            
            # If we are at the beginning of the word processing, print info about source text
            # Then info about each extracted word can be displayed
            if i == 0:
                print(f"\nTranslation Model: {self.transl_model_name}")
                if self.gen_model_name:
                    print(f"Sentence Generation Model: {self.gen_model_name}")
                if self.tts_model_name:
                    print(f"TTS Model: {self.tts_model_name}")
                print(f"Source language: {self.src_lang_fullname}")
                print(f"Target language: {self.target_lang_fullname}")
                # If the source language is Chinese, print the Pinyin of the source text
                if self.src_lang == "zh":
                    print(f"\n{yellow('Source Text:')} {self.src_text} ({self.src_text_pinyin})")
                else:
                    print(f"\n{yellow('Source Text:')} {self.src_text}")
                # If the target language is Chinese, print the Pinyin of the translated source text
                if self.target_lang == "zh":
                    print(f"{yellow('Translation:')} {self.full_translation} ({self.full_translation_pinyin})\n")
                else:
                    print(f"{yellow('Translation:')} {self.full_translation}\n")
            
            # Check if the source word was already part of the vocab list
            if row_indices:
                # Update existing word
                for row_index in row_indices:
                    # Get the entry (row) associated with a given row index
                    entry = csv_rows[row_index]
                    # Update the entry (existing word)
                    self._process_src_word(src_word, pos_tag, entry)
            else:
                # Process new word
                new_entry = init_entry_values.copy()
                new_entry['Word'] = src_word
                # Remove pinyin columns if necessary
                if self.src_lang != 'zh':
                    del new_entry['Pinyin (Word)']
                if self.target_lang != 'zh':
                    del new_entry['Pinyin (Translation)']
                    
                self._process_src_word(src_word, pos_tag, new_entry, is_new_entry=True)
                
                # Save new entry and source word
                new_entries.append(new_entry)
                new_words.append(src_word)
         
        # Save all new entries with the original vocal list
        csv_rows.extend(new_entries)
        
        return csv_rows
    
    def _process_src_word(self, src_word, pos_tag, entry, is_new_entry=False):
        """
        Process a source word, updating or adding it to the vocabulary list.

        Args:
            src_word (str): The source word to process.
            pos_tag (str): The part-of-speech tag for the source word.
            entry (dict): The vocabulary entry associated with the source word.
            is_new_entry (bool, optional): Indicates whether the entry is new. Defaults to False.
            
        Returns:
            None
        """
        updated = False
        
        # Do the translation of the source word only if one of these cases is true:
        # 1. It is a new entry
        # 2. The original translation needs to be updated because the current chosen translation model is different
        if not entry['Translation'] or ((entry['Translation'] and entry['Translation Model'] != self.transl_model_name)):
            updated = True
            # Translate the source word
            if src_word in COMMON_PARTICLES:
                # The source word is a Chinese particle. Get its translation directly from the map
                word_translation = COMMON_PARTICLES[src_word]
            else:
                word_translation = self.transl_model.translate_text(src_word)
                if self.src_lang == 'zh' and word_translation.lower() == "yes.":
                    word_translation = "like"  # Fixing the incorrect translation for "喜欢"
            # Save new translation
            entry['Translation'] = word_translation
            entry['Translation Model'] = self.transl_model_name
            
        # Save pinyin for the source word if the following are true:
        # - The source language is Chinese 
        # - The entry doesn't already have a pinyin for the source word
        if self.src_lang == 'zh' and not entry['Pinyin (Word)']:
            # Get Pinyin for the source word and save it
            entry['Pinyin (Word)'] = self._convert_to_pinyin(src_word)
            
        # Save pinyin for the translated word if following are true:
        # - The target language is Chinese 
        # - The entry doesn't already have a pinyin for the translated word
        if self.target_lang == 'zh':
            # Get Pinyin for the translated word and save it
            entry['Pinyin (Translation)'] = self._convert_to_pinyin(word_translation)
            
        # Add the origin of the word only if this condition is true:
        # 1. The corresponding flag is enabled
        if self.src_text_origin:
            updated = True
            entry['Word Origin'] = self.src_text_origin
            
        # Get the full name of the POS tag, e.g. a --> adjective, ad --> adverbial
        # TODO: fix pos full name (e.g. adj not covered)
        #pos_full_name = POS_MAP.get(pos_tag, 'unknown')
        # Add the POS only if these conditions are true:
        # 1. The corresponding flag is enabled
        # 2. The corresponding entry is empty
        if self.add_pos and not entry['POS']:
            entry['POS'] = pos_tag  # pos_full_name
            
        # Generate speech for the source word if:
        # 1 The corresponding flag is enabled
        #               AND
        # 2.1 The entry doesn't already have an audio path for the source word 
        #               OR
        # 2.2 The audio path doesn't exist
        if self.add_audio_words and (not entry['Audio (Word)'] or 
            not os.path.exists(entry['Audio (Word)'])):
            updated = True
            # Generate sound pronounciation for the source word as a background job
            # i.e. 'submit' event listener
            # TODO important: find another heuristic than `self.src_lang != 'zh'` ...
            if self.src_lang != 'zh' and len(src_word) == 1:
                job = self.tts_model.generate_speech(
                    text=f"{src_word}, {src_word}, {src_word}", lang=self.src_lang, event_listener="submit")
            else:
                job = self.tts_model.generate_speech(
                    text=src_word, lang=self.src_lang, event_listener="submit")
            
            # Get the source word or its pinyin
            if self.src_lang == "zh":
                # Get Pinyin for the source word
                pinyin_src_word = self._convert_to_pinyin(src_word)
                filename = pinyin_src_word
            else:
                filename = src_word
            
            # Filepath to the audio associated with the source word
            audio_path = os.path.join(self.audio_words_dirpath, filename +".wav")
            # Save the source word or its pinyin along with the 'job' for later processing
            self.audio_jobs.append((audio_path, job))
            # Save the audio path associated with the source word
            if self.audio_base_url:
                audio_words_dirpath = os.path.join(self.audio_base_url, f"{self.src_lang_fullname.lower()}/words")
                entry['Audio (Word)'] = "file://" + os.path.join(audio_words_dirpath, filename +".wav")
            else:
                entry['Audio (Word)'] = "file://" + audio_path
            entry['TTS Model'] = self.tts_model_name
            
        # Generate the example sentences only if one of these cases is true:
        # 1. The corresponding flag is enabled
        # 1. It is a new entry
        # 2. The original example sentences need to be updated because the current chosen generation model is different
        if self.add_sentences and (not entry['Example Sentences'] or (entry['Example Sentences'] and entry['Sentence Generation Model'] != self.gen_model_name)):
            updated = True
            sentences = self.gen_model.generate_text(words=src_word, nb_sentences=2).strip()
            
            # Save the sentence generation model
            entry['Example Sentences'] = sentences
            entry['Sentence Generation Model'] = self.gen_model_name
        
        # Add updated date only if it is not a new entry and an entry's value was updated
        if updated and not is_new_entry:
            entry['Date Updated'] = datetime.now().strftime("%Y-%m-%d")
        
        # Print info about the given source word including translation and pinyin (if it applies)
        # Only if it is a new word in the vocab list
        if is_new_entry:
            self._print_entry(entry) 
    
    def _save_csv_file(self, csv_rows, csv_filepath=""):
        """
        Save the translation data into a CSV file.

        Args:
            csv_rows (list): List of dictionaries representing rows of data to be saved in CSV format.
            csv_filepath (str, optional): Filepath where the CSV file will be saved. If not provided,
                a default filepath is used based on the source language. Defaults to "".
                
        Returns:
            0 if CSV data could be saved or 1 if csv_rows is empty.

        Notes:
            This method checks if pandas (pd) is imported; if not, it imports it dynamically.
            It then converts the list of dictionaries (csv_rows) into a pandas DataFrame
            and saves it as a CSV file. If no filepath is provided, a default filepath is used
            based on the source language.
        """
        def add_column_styles(df, column_styles, comments_editable=False):
            # Define styles for specific columns: column_styles

            # Create HTML for table headers
            th_elements = df.columns.map(lambda col: f'<th style="{column_styles.get(col, "")}">{col}</th>')

            # Create HTML for table rows
            tr_elements = df.apply(lambda row: ''.join([f'<td contenteditable="{True if comments_editable and col == "Comments" else False}" style="{column_styles.get(col, "")}">{row[col]}</td>' for col in df.columns]), axis=1)
            
            # Combine into complete HTML table
            html = f'<table id="myTable"><thead><tr>{" ".join(th_elements)}</tr></thead><tbody>'
            for row in tr_elements:
                html += f'<tr>{row}</tr>'
            html += '</tbody></table>'

            return html

        def make_clickable(val):
            return f'<a href="{val}">{os.path.basename(val)}</a>'

        def replace_newlines_and_bold(val):
            if isinstance(val, str):  # Check if the value is a string
                val = val.replace('\n', '<br>')  # Replace newline characters
                val = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', val)  # Replace **text** with <strong>text</strong>
            return val

        styles = '''
        <style>
        table {
          width: 100%;
          border-collapse: collapse;
        }
        th, td {
          padding: 12px;
          text-align: left;
          border: 1px solid #dddddd;
          white-space: nowrap; /* Prevent text wrapping */
          overflow: hidden; /* Hide overflow text */
          /*text-overflow: ellipsis; /* Show ellipsis for overflow text */
        }
        th {
          background-color: #f2f2f2;
        }
        tr:nth-child(even) {
          background-color: #f9f9f9;
        }
        </style>
        '''

        # Adding DataTables CSS and JS, and jQuery UI for resizing
        datatables_includes = '''
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/colreorder/1.5.4/js/dataTables.colReorder.min.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/colreorder/1.5.4/css/colReorder.dataTables.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/colresizable/1.6.0/colResizable-1.6.min.js"></script>
        <script>
        $(document).ready( function () {
            // Initialize DataTables with column reorder and resizing
            var table = $('#myTable').DataTable({
                colReorder: true,
                // Initial column widths
                // Adjust based on column index
                columnDefs: [
                    {column_defs} 
                ],
            });
       
            // Initialize column resizing
            $('#myTable').colResizable({
                liveDrag: true,
                postbackSafe: true,
                partialRefresh: true,
                resizeMode: 'fit',  // Optionally adjust resizing mode
                onResize: function(event) {
                    // Save column widths after resize (example: store in local storage)
                    localStorage.setItem('dataTableColumnWidths-' + tableIdentifier, JSON.stringify(table.columns().width().toArray()));
                }
            });
        
            // Optionally, restore column widths on page load
            var storedColumnWidths = localStorage.getItem('dataTableColumnWidths-' + tableIdentifier);
            if (storedColumnWidths) {
                table.columns().every(function(index) {
                    this.width(JSON.parse(storedColumnWidths)[index]);
                });
                table.draw();  // Redraw the table with restored widths
            }
            
            // Load comments from local storage
            {load_comments}
        } );
        </script>
        <script>
        function saveComments() {
            var comments = {};
            $('#myTable tbody tr').each(function() {
                var word = $(this).find('td').eq({word_index}).text();
                var comment = $(this).find('td').eq({comments_index}).text();
                comments[word] = comment;
            });
            localStorage.setItem('comments-' + tableIdentifier, JSON.stringify(comments));
            alert('Comments saved!');
        }

        function loadComments() {
            var comments = JSON.parse(localStorage.getItem('comments-' + tableIdentifier));
            if (comments) {
                $('#myTable tbody tr').each(function() {
                    var word = $(this).find('td').eq({word_index}).text();
                    if (comments[word]) {
                        $(this).find('td').eq({comments_index}).text(comments[word]);
                    }
                });
            }
        }
        </script>
        '''
        
        if len(csv_rows[0]) == 0:
            print("`csv_rows` is empty! No translation data to be saved in a CSV file.")
            return 1
        
        if self.pd is None:
            # import pandas as pd
            self.pd = import_module("pandas", alias="pd")
        
        # Create a DataFrame from the CSV rows
        df = self.pd.DataFrame(csv_rows)
        
        column_to_min_width = {}
        valid_column_names = list(csv_rows[0].keys())
        for col_name in valid_column_names:
            column_to_min_width[col_name] = COLUMN_TO_MIN_WIDTH[col_name]
            
        column_to_col_index = dict([(k, idx) for idx, (k,v) in enumerate(column_to_min_width.items())])
        
        column_defs = ""
        for idx, (col, min_width) in enumerate(column_to_min_width.items()):
            width = min_width.split(":")[1].strip()
            column_defs += f"{{ width: '{width}', targets: {idx} }},"

        datatables_includes = datatables_includes.replace("{word_index}", str(column_to_col_index['Word']))
        datatables_includes = datatables_includes.replace("{comments_index}", str(column_to_col_index['Comments']))
        datatables_includes = datatables_includes.replace("{column_defs}", column_defs)

        # If not filepath for CSV file given, save it in the current working directory
        if not csv_filepath:
            csv_filepath = os.path.join(
                os.getcwd(), 
                f'{self.src_lang_fullname.lower()}_words_{self.target_lang_fullname.lower()}_translation.csv')
        # File path of CSV file without extension
        filepath_no_ext = get_filepath_without_extension(csv_filepath)
     
        # Create a unique identifier for the table based on the current time
        tableIdentifier = re.sub(r'\W+', '', str(time.time()))  # Create a sanitized identifier

        # Save the translation data in a CSV file
        print(f"\nSaving CSV file: {csv_filepath}")
        df.to_csv(f'{csv_filepath}', index=False, encoding='utf-8-sig')

        ## HTML page containing the table
        # Apply the necessary transformations
        df = df.map(replace_newlines_and_bold)
        if 'Audio (Word)' in df.columns:
            df['Audio (Word)'] = df['Audio (Word)'].apply(make_clickable)
        
        if self.add_save_comments_button:
            load_comments = "loadComments();"
            save_comments_button = """
                <div style="position: relative; overflow-x: auto;">
                    <div style="position: fixed; top: 2px; right: 500px; z-index: 100;">
                        <button onclick="saveComments()" style="padding: 8px 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 4px; font-size: 14px;">
                            Save Comments
                        </button>
                    </div>
                </div>
            """
        else:
            load_comments = ""
            save_comments_button = ""
 
        datatables_includes = datatables_includes.replace("{load_comments}", load_comments)

        html_content = f"""
            <script>var tableIdentifier = "{tableIdentifier}";</script>
            {styles}
            {datatables_includes}
            {add_column_styles(df, column_styles=column_to_min_width, comments_editable=self.add_save_comments_button)}
            {save_comments_button}
        """
        
        # Save the HTML content to a file
        html_file_path = f'{filepath_no_ext}.html'  
        with open(html_file_path, 'w') as file:
            file.write(html_content)

        print(f"Saving HTML file: {html_file_path}")
        
        return 0
    
    def _segment_text(self, text, lang):
        """
        Segment text into individual words and their parts of speech.

        Args:
            text (str): The text to segment.
            lang (str): The language code of the text.

        Returns:
            list: A list of tuples containing segmented words and their parts of speech.
            
        Raises:
            ValueError: if the given language is not supported by spaCy.
        """
        if lang == 'zh':
            if self.pseg is None:
                # import jieba
                jieba = import_module("jieba")
                # import jieba.posseg as pseg
                self.pseg = import_with_importlib("jieba.posseg", alias="pseg")
            words = self.pseg.lcut(text)
            return list(words)  # Keeping order by using list instead of set
        else:
            spacy_model_name = f'{lang}_core_web_sm'
            if self.spacy is None:
                # import spacy
                self.spacy = import_module("spacy")
            print("Loading spaCy model...")
            nlp = self._load_spacy_model(spacy_model_name)
            if nlp == 1:
                print(f"spaCy model '{spacy_model_name}' not found. Trying to download it.")
                download_spacy_model(spacy_model_name)
                nlp = self._load_spacy_model(spacy_model_name)
                if nlp == 1:
                    print(f"spaCy model '{spacy_model_name}' couldn't be loaded. Trying to "
                          f"load another model: '{lang}_core_news_sm'")
                    spacy_model_name = f'{lang}_core_news_sm'
                    nlp = self._load_spacy_model(spacy_model_name)
                    if nlp == 1:
                        print(f"spaCy model '{spacy_model_name}' not found. Trying to download it.")
                        download_spacy_model(spacy_model_name)
                        nlp = self._load_spacy_model(spacy_model_name)
                        if nlp == 1:
                            raise ValueError(f"Unsupported language: {lang}")
                            
            print(f"spaCy model '{spacy_model_name}' was successfully loaded!")
            print("")

            doc = nlp(text)
            return [(token.text, token.pos_) for token in doc]
    
    def translate(self, src_text, target_lang, src_text_origin="", 
                  lang_detector=LANG_DETECTOR, 
                  transl_model_name=TRANSL_MODEL_MAP.get(TRANSL_MODEL), 
                  gen_model_name=GEN_MODEL_MAP.get(GEN_MODEL),
                  vocab_csv_filepath="", audio_main_dirpath=AUDIO_MAIN_DIRPATH, audio_base_url="",
                  add_pos=False, add_sentences=False, add_audio_text=False, add_audio_words=False,
                  add_save_comments_button=False):
        """
        Translate the source text into the target language and perform various augmentation tasks.

        Args:
            src_text (str): The source text to be translated.
            target_lang (str): The target language code to translate the text into.
            src_text_origin (str): Origin of the source text. Default to "".
            lang_detector (str, optional): The language detector to determine the source text's language. Defaults to "polyglot".
            transl_model_name (str, optional): The translation model to use. Defaults to "Helsinki-NLP/opus-mt".
            gen_model_name (str, optional): The text generation model to use. Defaults to "gemini-pro".
            vocab_csv_filepath (str, optional): Path to the CSV file to save the vocabulary. Defaults to "".
            audio_main_dirpath (str, optional): Path to the main directory for storing audio files. Defaults to "~/audio/".
            add_pos (bool, optional): Whether to add part-of-speech tags. Defaults to False.
            add_sentences (bool, optional): Whether to generate example sentences. Defaults to False.
            add_audio_text (bool, optional): Whether to generate audio for the entire text. Defaults to False.
            add_audio_words (bool, optional): Whether to generate audio for individual words. Defaults to False.

        Returns:
            int: 0 if successful, 1 if language detection fails.

        Raises:
            ValueError: If source language equals target language.

        Notes:
            This method performs the following tasks:
            1. Detects the language of the source text.
            2. Loads the necessary translation, text generation, and TTS models.
            3. Converts the source text to pinyin if it's in Chinese.
            4. Translates the source text into the target language.
            5. Processes each word extracted from the source text.
            6. Optionally generates example sentences.
            7. Saves the vocabulary to a CSV file.
            8. Generates audio files for the entire text and individual words if requested.
        """
        # TODO: find how to add default values in docstrings, e.g. lang_detector
        
        ## Dynamic Parameter Attributes
        # Get the frame object for the caller's stack frame
        frame = inspect.currentframe()
        # Get the arguments passed to the method
        args, _, _, values = inspect.getargvalues(frame)
        # Iterate over the arguments and set them as attributes
        for arg in args[1:]:  # Skip 'self'
            setattr(self, arg, values[arg])
        
        # Detect language of source text
        self.src_lang = detect_language(self.src_text, lang_detector=self.lang_detector)
        if self.src_lang is None:
            # Language detection failed
            return 1
        
        # TODO: explain why, you can have zh-... and zh-...
        if self.src_lang.startswith("zh"):
            self.src_lang = "zh"
    
        if self.src_lang == self.target_lang:
            raise ValueError("Source and target languages must be different!")
            
        # Get fullnames of languages from language codes
        self.src_lang_fullname = get_language_full_name(self.src_lang)
        self.target_lang_fullname = get_language_full_name(self.target_lang)
        
        # Remove punctuation from the source text
        # Segmentation done on text without punctuation
        # TODO: protegê-la -> protegêla
        self.src_text_no_punct = remove_punctuation(self.src_text)
        
        # Load the TTS model and create audio directories
        self._load_TTS_model()

        # Lad the right translation model
        self._load_transl_model()
            
        # Load the right text generation model
        self._load_gen_model()

        # Convert the source text into pinyin if it is necessary
        if self.src_lang == "zh":
            self.src_text_pinyin = self._convert_to_pinyin(self.src_text)
        else:
            self.src_text_pinyin = ""
                
        # Translate the entire text to get context-aware translation
        self.full_translation = self.transl_model.translate_text(self.src_text)
        # Convert the full translation into pinyin if it is the case
        if self.target_lang == "zh":
            self.full_translation_pinyin = self._convert_to_pinyin(self.full_translation)
        else:
            self.full_translation_pinyin = ""
        
        # Get the file paths of the audio and content of the source text
        if self.add_audio_text:
            if self.src_lang == "zh":
                filename = self.src_text_pinyin[:100].strip()
                self.audio_text_content = self.src_text + "\n\n" + self.src_text_pinyin
            else:
                filename = self.src_text_no_punct[:100].strip()
                self.audio_text_content = self.src_text
            self.audio_text_filepath = os.path.join(self.audio_text_dirpath, filename + ".wav")
            self.audio_text_content_filepath = os.path.join(self.audio_text_dirpath, filename + ".txt")

        # Segment the source text (with no punctuations) into words with their parts of speech
        # words is a list of tuples (word, POS), e.g. ('crucial', 'ADJ')
        words = self._segment_text(self.src_text_no_punct, self.src_lang)

        # Process each extracted word: translate, get its pinyin (if it applies), 
        # generate example sentences, generate pronounciation sounds, ...
        csv_rows = self._process_extracted_words(words)

        self._save_csv_file(csv_rows, self.vocab_csv_filepath)

        print("")
        
        ## Audio generation tasks: 
        # 1. Generate audio sounds for the whole source text
        # 2. Generate audio sounds for each extracted word from the source text
        # TODO: if audio couldn't be generated (e.g. audio file already present), display message
        self._process_audio_text()
        self._process_audio_words()
        
        print("")
        
        return 0


###############
# Main Function
###############
def main():
    exit_code = 0
    parser = setup_argparser()

    # Process arguments
    if not args.csv_filepath:
        args.csv_filepath = ""
        
    # Translate short model names to full model names
    args.transl_model = TRANSL_MODEL_MAP.get(args.transl_model, TRANSL_MODEL)
    args.gen_model = GEN_MODEL_MAP.get(args.gen_model, GEN_MODEL)
    
    vocab_aug = VocabAugmentor()
    vocab_aug.translate(args.text, 
                        args.target_lang, 
                        args.text_origin,
                        lang_detector=args.lang_detector, 
                        transl_model_name=args.transl_model, 
                        gen_model_name=args.gen_model, 
                        vocab_csv_filepath=args.csv_filepath, 
                        audio_main_dirpath=args.audio_dirpath,
                        audio_base_url=args.audio_base_url,
                        add_pos=args.add_pos, 
                        add_sentences=args.add_sentences, 
                        add_audio_text=args.add_audio_text, 
                        add_audio_words=args.add_audio_words,
                        add_save_comments_button=args.add_save_comments_button)
    return exit_code


if __name__ == '__main__':
    retcode = main()
    print(f'Program exited with {retcode}')
