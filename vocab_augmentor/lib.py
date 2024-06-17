#########################
## Default Config Options
#########################
AUDIO_MAIN_DIRPATH = "~/audio/"

GEN_MODEL = 'gemini'
LANG_DETECTOR = 'polyglot'
TRANSL_MODEL = 'helsinki'

LANG_DETECTORS = ['langdetect', 'langid', 'polyglot']


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
