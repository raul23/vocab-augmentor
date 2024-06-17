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
