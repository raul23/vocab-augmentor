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

from datetime import datetime

# Suppress the specific warning by setting the logging level to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

UDIO_MAIN_DIRPATH = "~/audio/"
GEN_MODEL = 'gemini'
LANG_DETECTOR = 'polyglot'
TRANSL_MODEL = 'helsinki'

LANG_DETECTORS = ['langdetect', 'langid', 'polyglot']

## Mapping of short model names to full model names
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

## Table columns info
COLUMN_TO_MIN_WIDTH = {
    'Word': 'min-width: 120px;', 
    'Pinyin (Word)': 'min-width: 120px;', 
    'Audio (Word)': 'min-width: 120px;',  
    'Translation': 'min-width: 120px;',
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

COLUMN_TO_COL_INDEX = dict([(k, idx) for idx, (k,v) in enumerate(COLUMN_TO_MIN_WIDTH.items())])

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


class LanguageModel:
    def __init__(self, src_lang, target_lang, model_name):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_fullname = get_language_full_name(src_lang)
        self.target_lang_fullname = get_language_full_name(target_lang)
        self.model_name = model_name
        
    def get_text_generation_request(self, words, nb_sentences):
        request = f"""
        Generate {nb_sentences} sentences in {self.src_lang_fullname} with the words '{words}' and
        provide their translations in {self.target_lang_fullname}. Use the following template:
        
        add text for sentence 1 here
        add translation for sentence 1 here
        
        add text for sentence 2 here
        add translation for sentence 2 here
        """
        return request
    
    def generate_text(self, words, nb_sentences=2):
        raise NotImplementedError("generate_text() not implemented!")
    
    def translate_text(self):
        raise NotImplementedError("translate_text() not implemented!")
        

# facebook/mbart-large-50-many-to-many-mmt
class MBart(LanguageModel):
    def __init__(self, src_lang, target_lang):
        super().__init__(src_lang, target_lang, 
                         model_name="facebook/mbart-large-50-many-to-many-mmt",)
        # Initialize mBART model and tokenizer
        # import transformers
        transformers = import_module("transformers")
        # from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        MBartForConditionalGeneration = import_module("transformers", "MBartForConditionalGeneration")
        MBart50TokenizerFast = import_module("transformers", "MBart50TokenizerFast")
        self.model  = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.tokenizer.src_lang = FACEBOOK_LANGUAGES[src_lang]
    
    def translate_text(self, text, return_tensors="pt", skip_special_tokens=True):
        encoded_text = self.tokenizer(text, return_tensors=return_tensors)
        generated_tokens = self.model.generate(
            **encoded_text,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[FACEBOOK_LANGUAGES[self.target_lang]]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=skip_special_tokens)[0]
    

# Helsinki-NLP/opus-mt
class HelsinkiNLP(LanguageModel):
    def __init__(self, src_lang, target_lang):
        super().__init__(src_lang, target_lang, model_name="Helsinki-NLP/opus-mt-itc-en")
                        # model_name=f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}")
        # Initialize the translation pipeline
        
        # import transformers
        transformers = import_module("transformers")
        # from transformers import pipeline
        pipeline = import_module("transformers", "pipeline")
        self.translator = pipeline("translation", model=self.model_name)
    
    def translate_text(self, text, max_length=512):
        translated = self.translator(text, max_length=max_length)
        return translated[0]['translation_text']
    

# gemini-pro
class GeminiPro(LanguageModel):
    def __init__(self, src_lang, target_lang):
        super().__init__(src_lang, target_lang, model_name="gemini-pro")
        # Configure the Gemini model
        api_key = get_api_key("GEMINI_API_KEY")
        # import google.generativeai as genai
        genai = import_with_importlib("google.generativeai", alias="genai")
        genai.configure(api_key = api_key)
        del api_key
        genmodel = genai.GenerativeModel(model_name=self.model_name)
        self.chat = genmodel.start_chat()
        
    def _send_message(self, request):
        # TODO: todo
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
        request = self.get_text_generation_request(words, nb_sentences)
        return self._send_message(request)
    
    def translate_text(self, text):
        return self._send_message(f"Translate this into {self.target_lang_fullname}: {text}")
    
    
# meta-llama/Meta-Llama-3-8B-Instruct
class Llama3(LanguageModel):
    def __init__(self, src_lang, target_lang):
        super().__init__(src_lang, target_lang, 
                         model_name="meta-llama/Meta-Llama-3-8B-Instruct")
        
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
        
        # Setup Pipeline and Tokenizer
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
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            pad_token_id=transformers.AutoTokenizer.from_pretrained(model_name).eos_token_id
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
        request = self.get_text_generation_request(words, nb_sentences)
        return self._gen_text(request, context, max_new_tokens, do_sample, temperature, top_p)
        
    
    def translate_text(self, text, context=None, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9):
        # Create the initial message with the user's request
        request = f"Provide just the answer (a single one) by translating the following into {self.target_lang_fullname}: {text}"

        return self._gen_text(request, context, max_new_tokens, do_sample, temperature, top_p)


class FacebookTTS:
    def __init__(self, model_name, verbose=False):
        # from transformers import VitsModel, AutoTokenizer
        VitsModel = import_module("transformers", "VitsModel")
        AutoTokenizer = import_module("transformers", "AutoTokenizer")
        # import torch
        self.torch = import_module("torch")
        # import scipy
        self.scipy = import_module("scipy")
        # import numpy
        self.np = import_module("numpy", alias="np")
        
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_speech(self, text, **kwargs): 
        inputs = self.tokenizer(text, return_tensors="pt")
        with self.torch.no_grad():
            output = self.model(**inputs).waveform
        audio_path = get_random_temp_filename(".wav")
        # Convert the PyTorch tensor to a NumPy array
        output_np = output.cpu().numpy()
        # Ensure the data is in the correct shape (1D array)
        if len(output_np.shape) > 1:
            output_np = output_np.squeeze()
        # Write the NumPy array to a WAV file
        self.scipy.io.wavfile.write(audio_path, rate=self.model.config.sampling_rate, data=output_np)
        return audio_path
       
    
class MeloTTS:
    def __init__(self, verbose=False):
        # from gradio_client import Client
        Client = import_module("gradio_client", "Client")
        self.client = Client("mrfakename/MeloTTS", verbose=verbose)
    
    def generate_speech(self, text, lang, event_listener="predict"):
        if lang == "en":
            speaker = "EN-Default"
        else:
            speaker = lang.upper()
            
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

    
def color(msg, msg_color='y', bold_msg=False):
    msg_color = msg_color.lower()
    colors = list(_COLOR_TO_CODE.keys())
    assert msg_color in colors, f'Wrong color: {msg_color}. Only these ' \
                                f'colors are supported: {msg_color}'
    msg = bold(msg) if bold_msg else msg
    msg = msg.replace(COLORS['NC'], COLORS['NC']+_COLOR_TO_CODE[msg_color])
    return f"{_COLOR_TO_CODE[msg_color]}{msg}{COLORS['NC']}"


def blue(msg):
    return color(msg, 'b')


def bold(msg):
    return color(msg, 'bold')


def green(msg):
    return color(msg, 'g')


def red(msg):
    return color(msg, 'r')


def violet(msg):
    return color(msg, 'v')


def yellow(msg):
    return color(msg)


def get_default_message(default_value):
    return green(f' (default: {default_value})')


def create_text_file(content, path):
    """
    Creates a text file with the given content at the specified path.
    
    Parameters:
    content (str): The content to be written to the text file.
    path (str): The path where the text file will be saved.
    
    Returns:
    None
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
    if lang_detector == "langdetect":
        return detect_language_with_langdetect(text)
    elif lang_detector == "langid":
        return detect_language_with_langid(text)
    elif lang_detector == "polyglot":
        return detect_language_with_polyglot(text)
    else:
        raise ValueError(f"Unsupported language detector: {lang_detector}")
    

def detect_language_with_langdetect(text):
    try:
        # Detect the language
        # Will return zh-cn, zh-tw (no 2-letters code), see https://pypi.org/project/langdetect/
        language_code = detect(text)
        return language_code
    except LangDetectException as e:
        # Handle cases where the language could not be detected
        print(f"Language detection failed: {e}")
        return None


def detect_language_with_langid(text):
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
    print("Downloading spaCy model...")
    # Suppress output by redirecting to os.devnull
    with open(os.devnull, 'w') as fnull:
        subprocess.run(
            ['python', '-m', 'spacy', 'download', model_name],
            stdout=fnull,
            stderr=fnull
        )


def get_api_key(token_name):
    api_key = os.environ.get(token_name)
    if api_key is None:
        print(f"Couldn't find {token_name} in environment variables")
        print(f"Reading {token_name} from Kaggle Secrets...")
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret(token_name)
    return api_key
        
        
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

    
def get_random_temp_filename(suffix=""):
    """
    Generates a random temporary filename without creating the file.
    
    Returns:
    str: The generated temporary filename.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            temp_file_path = temp_file.name
        #print(f"Generated temporary filename: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def import_module(module_name, attribute_name=None, alias=None):
    try:
        module = import_with_importlib(module_name, attribute_name, alias)
    except ImportError:
        print(f"{module_name} module not found. Installing...")
        install(module_name)
        module = import_with_importlib(module_name, attribute_name, alias)
    return module
    

def import_with_importlib(module_name, attribute_name=None, alias=None):
    """Dynamically import a module or an attribute from a module using importlib.
    
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
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("\n")

    
def read_csv(filename):
    """Reads the CSV file and returns a list of dictionaries."""
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


def move_file(source_path, destination_dir, destination_filename):
    """
    Moves a file from source_path to destination_dir with a new name destination_filename.
    
    Parameters:
    source_path (str): The path to the source file.
    destination_dir (str): The path to the destination directory.
    destination_filename (str): The new name for the file at the destination.
    
    Returns:
    str: The path to the moved file.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Destination path including the new file name
    destination_path = os.path.join(destination_dir, destination_filename)
    
    # Move the file
    shutil.move(source_path, destination_path)
    
    return destination_path


def rename_file(file_path, new_file_name):
    # Get the directory and original file name
    directory, original_file_name = os.path.split(file_path)
    
    # Create the full path for the new file
    new_file_path = os.path.join(directory, new_file_name)
    
    # Rename the file
    os.rename(file_path, new_file_path)
    
    # print(f'Renamed file: {file_path} -> {new_file_path}')
    return new_file_path


def write_csv(filename, data, fieldnames):
    """Writes the data to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


class VocabAugmentor:
    
    def __init__(self):
        self.pd = None  # pandas
        self.pinyin = None
        self.pseg = None  # For segmentation of Chinese text
        self.spacy = None  # For segmentation of non-Chinese text
    
    def _convert_to_pinyin(self, word):
        """
        Convert a Chinese word to its pinyin representation.

        Args:
            word (str): The Chinese word to convert.

        Returns:
            str: The pinyin representation of the word.
        """
        if self.pinyin is None:
            # from pypinyin import pinyin
            self.pinyin = import_module("pypinyin", "pinyin")
        pinyin_list = self.pinyin(word)
        return ' '.join([syllable[0] for syllable in pinyin_list])
    
    def _is_language_supported_by_tts(self, lang):
        melotts_supported_languages = ['EN', 'ES', 'FR', 'ZH', 'JP', 'KR']
        facebook_tts = ['PT']
        all_supported_languages = melotts_supported_languages + facebook_tts
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
            print("Loading spaCy model...")
            nlp = self.spacy.load(model_name)
        except OSError:
            return 1
        return nlp
    
    def _segment_text(self, text, lang):
        """
        Segment text into individual words and their parts of speech.

        Args:
            text (str): The text to segment.
            lang (str): The language of the text.

        Returns:
            list: A list of tuples containing segmented words and their parts of speech.
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
            nlp = self._load_spacy_model(spacy_model_name)
            if nlp == 1:
                print(f"spaCy model '{spacy_model_name}' not found. Trying to download it.")
                download_spacy_model(spacy_model_name)
                nlp = self._load_spacy_model(spacy_model_name)
                if nlp == 1:
                    print(f"spaCy model '{spacy_model_name}' not found. Trying to "
                          f"load another model: '{lang}_core_news_sm'")
                    spacy_model_name = f'{lang}_core_news_sm'
                    nlp = self._load_spacy_model(spacy_model_name)
                    if nlp == 1:
                        print(f"spaCy model '{spacy_model_name}' not found. Trying to download it.")
                        download_spacy_model(spacy_model_name)
                        nlp = self._load_spacy_model(spacy_model_name)
                        if nlp == 1:
                            raise ValueError(f"Unsupported language: {lang}")
            print("")

            doc = nlp(text)
            return [(token.text, token.pos_) for token in doc]

    def translate(self, src_text, target_lang, transl_model_name="Helsinki-NLP/opus-mt", gen_model_name="gemini-pro", 
                  lang_detector="polyglot", vocab_csv_file=None, add_pos=False, add_sentences=False, 
                  add_audio_text=False, add_audio_words=False):
        """
        Translate text from a source language to a target language using a specified model.

        Args:
            src_text (str): The source text to translate.
            target_lang (str): The target language code (e.g., 'zh' for Chinese).
            transl_model_name (str): The name of the translation model to use.
            gen_model_name (str): The name of the sentence generation model to use.
            vocab_list (list, optional): ...
            add_pos (bool, optional): Whether to add parts of speech to the output. Defaults to False.
            add_sentences (bool, optional): Whether to add sentences in the source language. Defaults to False.
            add_audio (bool, optional): ...

        Raises:
            ValueError: If the source and target languages are the same or if an unsupported model is provided.

        Returns:
            
        """
        src_lang = detect_language(src_text, lang_detector="polyglot")
        if src_lang is None:
            return 1
        # TODO: explain why, you can have zh-... and zh-...
        if src_lang.startswith("zh"):
            src_lang = "zh"
        if src_lang == target_lang:
            raise ValueError("Source and target languages must be different!")
            
        src_lang_fullname = get_language_full_name(src_lang)
        target_lang_fullname = get_language_full_name(target_lang)
        
        if add_audio_words or add_audio_text:
            if self._is_language_supported_by_tts(src_lang):
                if src_lang == "pt":
                    tts_model = FacebookTTS("facebook/mms-tts-por")
                else:
                    tts_model = MeloTTS()
            else:
                add_audio_words = False
                add_audio_text = False
            if add_audio_words:
                # Create the audio directory for words
                words_audio_dir = f"/kaggle/working/audio/{src_lang_fullname.lower()}/words"
                os.makedirs(words_audio_dir, exist_ok=True)
            if add_audio_text:
                # Create the audio directory for text
                text_audio_dir = f"/kaggle/working/audio/{src_lang_fullname.lower()}/text"
                os.makedirs(text_audio_dir, exist_ok=True)

        if transl_model_name == "Helsinki-NLP/opus-mt":
            transl_model = HelsinkiNLP(src_lang, target_lang)
        elif transl_model_name == "facebook/mbart-large-50-many-to-many-mmt":
            transl_model = MBart(src_lang, target_lang)
        elif transl_model_name == "gemini-pro":
            transl_model = GeminiPro(src_lang, target_lang)
        elif transl_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            transl_model = Llama3(src_lang, target_lang)
        else:
            raise ValueError(f"Unsupported translatation model: {transl_model_name}")
            
        if add_sentences:
            if gen_model_name == "gemini-pro":
                if transl_model_name == "gemini-pro":
                    gen_model = transl_model
                else:
                    gen_model = GeminiPro(src_lang, target_lang)
            elif gen_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                if transl_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                    gen_model = transl_model
                else:
                    gen_model = Llama3(src_lang, target_lang)
            else:
                raise ValueError(f"Unsupported sentence generation model: {gen_model_name}")

        if src_lang == "zh":
            src_text_pinyin = self._convert_to_pinyin(src_text)
        else:
            src_text_pinyin = ""
                
        # Translate the entire text to get context-aware translation
        full_translation = transl_model.translate_text(src_text)
        if target_lang == "zh":
            full_translation_pinyin = self._convert_to_pinyin(full_translation)
        else:
            full_translation_pinyin = ""
            
        # Remove punctuation from the source text
        # TODO: copy src_text and save it in txt file (content) when adding text audio
        src_text = remove_punctuation(src_text)

        # Segment the source text into words with their parts of speech
        words = self._segment_text(src_text, src_lang)

        # Create a list to store the word details
        word_details = []

        # Print the vocabulary list with translations and pinyin (if it applies)
        index = 0
        sentences_to_print = ""
        audio_jobs = []
        chinese_chars_to_pinyin ={}
        for src_word, flag in words:
            if src_word in COMMON_PARTICLES:
                translation = COMMON_PARTICLES[src_word]
            else:
                translation = transl_model.translate_text(src_word)
                if src_lang == 'zh' and translation.lower() == "yes.":
                    translation = "like"  # Fixing the incorrect translation for "喜欢"
            pos_full_name = POS_MAP.get(flag, 'unknown')
            word_detail = {
                f'{src_lang_fullname} Word': src_word,
            }
            word_detail_str = src_word
            if src_lang == 'zh':
                pinyin_src_word = chinese_chars_to_pinyin.get(src_word)
                if pinyin_src_word is None:
                    pinyin_src_word = self._convert_to_pinyin(src_word)
                    chinese_chars_to_pinyin.setdefault(src_word, pinyin_src_word)
                word_detail.update({'Pinyin': pinyin_src_word})
                word_detail_str += f" ({pinyin_src_word})"
            if add_audio_words:
                job = tts_model.generate_speech(text=src_word, lang=src_lang, event_listener="submit")
                if src_lang == "zh":
                    pinyin_src_word = chinese_chars_to_pinyin.get(src_word)
                    if pinyin_src_word is None: 
                        pinyin_src_word = self._convert_to_pinyin(src_word)
                        chinese_chars_to_pinyin.setdefault(src_word, pinyin_src_word)
                    audio_jobs.append((pinyin_src_word, job))
                else:
                    audio_jobs.append((src_word, job))
            word_detail.update({f'{target_lang_fullname} Translation': translation})
            word_detail_str += f": {translation} "
            if target_lang == 'zh':
                pinyin_translated_word = chinese_chars_to_pinyin.get(translation)
                if pinyin_translated_word is None: 
                    pinyin_translated_word = self._convert_to_pinyin(translation)
                    chinese_chars_to_pinyin.setdefault(translation, pinyin_translated_word)
                word_detail.update({'Pinyin': pinyin_translated_word})
                word_detail_str += f"({pinyin_translated_word}) "
            if add_pos:
                word_detail.update({'Part of Speech': pos_full_name})
                word_detail_str += f"- {pos_full_name}"
            if add_sentences:
                sentences = gen_model.generate_text(words=src_word, nb_sentences=2).strip()
                #ipdb.set_trace()
                try:
                    sent1, transl1, _, sent2, transl2 = sentences.split("\n")
                except:
                    ipdb.set_trace()
                if src_lang == 'zh':
                    sent_pinyin_transl = f"1. {sent1} ({self._convert_to_pinyin(sent1)}): {transl1}\n2. {sent2} ({self._convert_to_pinyin(sent2)}): {transl2}"
                    sentences_to_print += f"\n{src_word} ({pinyin_src_word}):\n{sent_pinyin_transl}\n"
                elif target_lang == 'zh':
                    sent_pinyin_transl = f"1. {sent1}: {transl1} ({self._convert_to_pinyin(transl1)})\n2. {sent2}: {transl2} ({self._convert_to_pinyin(transl2)})"
                    sentences_to_print += f"\n{src_word}:\n{sent_pinyin_transl}\n"
                else:
                    sent_pinyin_transl = f"1. {sent1}: {transl1}\n2. {sent2}: {transl2}"
                    sentences_to_print += f"\n{src_word.capitalize()}:\n{sent_pinyin_transl}\n"
                word_detail.update({'Examples of sentences': sent_pinyin_transl})
            word_details.append(word_detail)
            if index == 0:
                print(f"\nTranslation Model: {transl_model_name}")
                print(f"Source language: {src_lang_fullname}")
                print(f"Target language: {target_lang_fullname}")
                if src_lang == "zh":
                    print(f"\nSource Text: {src_text} ({src_text_pinyin})")
                else:
                    print(f"\nSource Text: {src_text}")
                if target_lang == "zh":
                    print(f"Full Translation: {full_translation} ({full_translation_pinyin})\n")
                else:
                    print(f"Full Translation: {full_translation}\n")
            index+=1
            print(word_detail_str)
        
        if add_sentences:
            print(f"\n\nExamples of sentences:")
            print(sentences_to_print)

        # Create a DataFrame from the word details
        if vocab_csv_file:
            # TODO: add translation model name, tts model name, date created, date updated
            if self.pd is None:
                # import pandas as pd
                self.pd = import_module("pandas", alias="pd")
            df = sef.pd.DataFrame(word_details)
            # Save the DataFrame to a CSV file
            filename = f'{src_lang_fullname.lower()}_words_translation.csv'
            print(f"\nSaving csv file: {filename}")
            df.to_csv(f'{filename}', index=False, encoding='utf-8-sig')

        print("")
        if add_audio_text:
            source_path = tts_model.generate_speech(text=src_text, lang=src_lang, event_listener="predict")
            if src_lang == "zh":
                filename = src_text_pinyin[:100].strip()
                content = src_text + "\n\n" + src_text_pinyin
            else:
                filename = src_text[:100].strip()
                content = src_text
            text_path = create_text_file(content, os.path.join(text_audio_dir, filename + ".txt"))
            audio_path = move_file(source_path, text_audio_dir, filename + ".wav")
            print(f"Text file: {text_path}")
            print(f"Audio file: {audio_path}")
        
        if add_audio_words:
            for word, job in audio_jobs:
                if type(job) == str:
                    source_path = job
                else:
                    source_path = job.result()
                # ipdb.set_trace()
                moved_file_path = move_file(source_path, words_audio_dir, word +".wav")
                print(f"File moved to {moved_file_path}")
        return 0


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        print(self.format_usage().splitlines()[0])
        self.exit(2, red(f'\nerror: {message}\n'))


class MyFormatter(argparse.HelpFormatter):
    """
    Corrected _max_action_length for the indenting of subactions
    """

    def add_argument(self, action):
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
    name_input = 'csv_file'
    try:
        width = os.get_terminal_size().columns - 5
        usage_msg = f'%(prog)s [OPTIONS] {{{name_input}}}'
    except OSError:
        # No access to terminal
        width = 110
        usage_msg = f'vocab [OPTIONS] {{{name_input}}}'
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
        name_input, nargs='*', action=required_length(0, 1),
        help='Path to the vocabulary CSV file. If the file does not exist, a new one will be created.')
    parser.add_argument(
        '-t', '--text', type=str, required=True, 
        help='The source text that will be processed to identify and translate new words.')
    parser.add_argument(
        '-l', '--target_lang', metavar='LANG_CODE', type=str, required=True, 
        help='''Target language code into which the source text will be translated (e.g., zh for Chinese, en for English,
             pt for Portuguese).''')
    parser.add_argument(
        '-m', '--transl_model', metavar='NAME', type=str, default=Transl_Model, 
        choices=['helsinki', 'mbart', 'gemini', 'llama'],
        help='Translation model to use for translating the text.'
             + get_default_message(Transl_Model))
    parser.add_argument(
        '-d', '--lang_detector', metavar='NAME', type=str, default=Lang_Detector, 
        choices=['langdetect', 'langid', 'polyglot'],
        help='Method to use for detecting the language of the source text.' 
             + get_default_message(Lang_Detector))
    parser.add_argument(
        '-g', '--gen_model', metavar='NAME', type=str, default=Gen_Model,
        choices=['gemini', 'llama'],
        help='Language model to use for generating example sentences in the source language.'
             + get_default_message(Gen_Model))
    parser.add_argument(
        '--as', '--add_sentences',  dest='add_sentences', action='store_true', 
        help='Flag to add or update example sentences in the vocabulary list.')
    parser.add_argument(
        '--ap', '--add_pos', dest='add_pos', action='store_true', 
        help='Flag to add or update part-of-speech (POS) information for the words.')
    parser.add_argument(
        '--aut', '--add_audio_text', dest='add_audio_text', action='store_true', 
        help='Flag to add or update audio pronunciation for the source text.')
    parser.add_argument(
        '--aaw', '--add_audio_words', dest='add_audio_words', action='store_true', 
        help='Flag to add or update audio pronunciation for the extracted words from the text.')     
    return parser


def main():
    exit_code = 0
    parser = setup_argparser()
    # TODO: uncomment when ready
    # args = parser.parse_args()
    #print(parser.print_help())
    #return 0
    
    args = parser.parse_args(['-t', 'É crucial protegê-la para o bem do planeta e das gerações futuras.', 
                              '-l', 'en', '-m', 'gemini'])
    
    # Process arguments
    if not args.csv_file:
        args.csv_file = ""
    # Translate short model names to full model names
    args.transl_model = MODEL_MAP.get(args.transl_model, args.transl_model)
    args.gen_model = MODEL_MAP.get(args.gen_model, args.gen_model)
    
    vocab_aug = VocabAugmentor()
    vocab_aug.translate(args.text, args.target_lang, transl_model_name=args.transl_model, gen_model_name=args.gen_model,
                        lang_detector=args.lang_detector, vocab_csv_file=args.csv_file, 
                        add_sentences=args.add_sentences, add_pos=args.add_pos, 
                        add_audio_words=args.add_audio_words, add_audio_text=args.add_audio_text)
    return exit_code


if __name__ == '__main__':
    retcode = main()
    print(f'Program exited with {retcode}')
