# Vocab-Augmentor
## Overview

**vocab** is a Python script designed to help language learners expand their 
vocabulary effortlessly. By leveraging advanced language models such as
`facebook/mbart-large-50-many-to-many-mmt`, `Helsinki-NLP/opus-mt`,
`Gemini 1.0 Pro`, and `llama 3-8b`, this tool identifies new words from any
given text, translates them, and updates your personal vocabulary list.
Additionally, it supports adding pronunciation sounds for new words using the
`MeloTTS` and `facebook/mms-tts-por` text-to-speech libraries for supported
languages.

![Table (HTML) displaying translation data including audio links, POS](./docs/images/table_sol_y_cielo.png)

## Features

- **Multi-Language Support**: Supports a variety of languages, making it versatile
  for learners of different languages.
- **Advanced Translation Models**: Choose from multiple state-of-the-art translation
  models for accurate translations.
  - `facebook/mbart-large-50-many-to-many-mmt`
  - `Helsinki-NLP/opus-mt`
  - `Gemini 1.0 Pro`
  - `llama 3-8b`
- **Automatic Segmentation**: Segments the provided text to identify individual words.
- **New Vocabulary Detection**: Detects words not present in your existing vocabulary
  list.
- **Detailed Word Information**: Saves new words along with their translation, pinyin
  (if applicable), and part of speech (POS).
- **Audio Pronunciation**: Adds pronunciation sounds for new words using `MeloTTS` for
  supported languages, including English, French, Spanish, Chinese, Korean, and Japanese.

  For Portuguese text-to-speech, `facebook/mms-tts-por` is used.
- **Selective Module Installation**: Modules are installed only when needed for
  specific tasks.
- **CSV and HTML Export**: Updates and exports the vocabulary table to a CSV file
  and a feature-rich HTML file.
  - HTML features include `DataTables` stylesheet, column reordering, resizable
    columns, search functionality, pagination, and sortable columns.

## Dependencies

### Requirements

- **Python**: version 3.10.3+
- `langcodes`: to get the full name of a language from its short code

### Selective Module Installation

Modules are installed automatically by the `vocab` script only when needed for specific 
tasks. Below is a breakdown of the modules required based on different functionalities:

1. **Translation Models**:
   - **facebook/mbart-large-50-many-to-many-mmt**:
     - `transformers`
   - **Helsinki-NLP/opus-mt**:
     - `transformers`

2. **Translation and Sentence Generation Models**:
   - **gemini-pro**:
     - `google.generativeai` (for API connection using *GEMINI_API_KEY*)
   - **meta-llama/Meta-Llama-3-8B-Instruct**:
     - `huggingface_hub` (to login and download **llama 3-8b**)
     - `torch`
     - `transformers`

3. **Text-to-Speech (TTS) Models**:
   - **facebook/mms-tts-por**:
     - `transformers`
     - `torch`
     - `scipy`
     - `numpy`
   - **MeloTTS**:
     - `gradio_client` (for API access to interact with **MeloTTS**)

4. **Language Detection Methods**:
   - **langdetect**:
     - `langdetect`
   - **langid**:
     - `langid`
   - **polyglot**:
     - `pyicu`
     - `pycld2`
     - `pycld3`

     Note: When using GPU T4 with **polyglot**, `pycld3` can't be
     installed, hence **polyglot** can't be used. Use an alternative
     detection method in such cases.

5. **Language-Specific Modules**:
   - **Chinese (source or target language)**:
     - `pypinyin`

6. **Data Management**:
   - **Saving translation data to CSV**:
     - `pandas`

7. **Text Segmentation**:
   - **Chinese**:
     - `jieba`
   - **Other languages**:
     - `spacy`

## Installation ⭐

Install the package `vocab_augmentor` with `pip`:
```terminal
pip install git+https://github.com/raul23/vocab-augmentor/#egg=vocab-augmentor
```

**Test installation**

1. Test your installation by importing `vocab_augmentor` and printing its version:
   ```terminal
   python -c "import vocab_augmentor; print(vocab_augmentor.__version__)"
   ```
2. You can also test that you have access to the `vocab` script by
   showing the program's version:
   ```terminal
   vocab --version
   ```
    
## API Key Management

To utilize the advanced translation and text generation features of
`Gemini 1.0 Pro` and `llama 3-8b`, API keys are required. These keys must be
saved as environment variables. Follow the steps below to manage your API keys:

### Setting Up Environment Variables

1. **Gemini 1.0 Pro API Key**
   - Save your Gemini 1.0 Pro API key as an environment variable named `GEMINI_API_KEY`.
   - In a Unix-based system (Linux, macOS), add the following line to your `.bashrc`
     or `.zshrc` file:
     ```sh
     export GEMINI_API_KEY='your_gemini_api_key'
     ```
   - In Windows, set the environment variable through the Command Prompt or System
     Properties:
     ```cmd
     setx GEMINI_API_KEY "your_gemini_api_key"
     ```

2. **Hugging Face API Key for Llama 3-8b**
   - Save your Hugging Face API key as an environment variable named `HF_API_KEY`.
   - In a Unix-based system, add the following line to your `.bashrc` or `.zshrc` file:
     ```sh
     export HF_API_KEY='your_hugging_face_api_key'
     ```
   - In Windows, set the environment variable through the Command Prompt or System
     Properties:
     ```cmd
     setx HF_API_KEY "your_hugging_face_api_key"
     ```

### Important Notes

- Ensure your API keys are kept confidential and not shared publicly.
- Using API keys allows the script to access powerful language models and 
  generate accurate translations and text examples efficiently.

By following these steps, you can seamlessly integrate API keys into the 
Vocab-Augmentor script and leverage its full capabilities for advanced language 
learning tasks.

## Performance Recommendations
    
- **GPU Recommendation**: When using `llama 3-8b`, GPU usage is highly recommended 
  for faster processing.
  
- **Text Generation**: Use either `Gemini 1.0 Pro` or `llama 3-8b` to generate
  example sentences.
  - `Gemini 1.0 Pro` is faster as it uses an API.

## Usage

### Script options ⭐

To display the `vocab` script list of options and their descriptions:
```terminal
usage: vocab [OPTIONS] {TEXT} {TGT_LANG}

Expand your vocabulary list by identifying and translating new words from provided text using various language models.

options:
  -v, --version                       Show program's version number and
                                      exit.
  -h, --help                          Display detailed usage instructions
                                      and exit the program.
  -t, --text TEXT                     The source text that will be
                                      processed to identify and translate
                                      new words.
  -l, --target_lang TGT_LANG          Target language code into which the
                                      source text will be translated (e.g.,
                                      zh for Chinese, en for English, pt
                                      for Portuguese).
  -o, --text_origin ORIGIN            Origin of the source text, e.g. movie
                                      script, book, URL of website, etc.
  -d, --lang_detector NAME            Method to use for detecting the
                                      language of the source text.
                                      (default: polyglot)
  -m, --transl_model NAME             Translation model to use for
                                      translating the text.
                                      (default: helsinki)
  -g, --gen_model NAME                Language model to use for generating
                                      example sentences in the source
                                      language. (default:
                                      gemini)
  -c, --csv_filepath CSV_FILE         Path to the vocabulary CSV file. If
                                      the file does not exist, a new one
                                      will be created.
  -a, --audio_dirpath AUDIO_DIR       Path to the main directory for
                                      storing audio files. (default:
                                      ~/audio/)
  -b, --audio_base_url URL            Base URL to audio files of words.
                                      (experimental)
  --ap, --add_pos                     Flag to add or update part-of-speech
                                      (POS) information for the words.
  --as, --add_sentences               Flag to add or update example
                                      sentences in the vocabulary list.
  --aut, --add_audio_text             Flag to add or update audio
                                      pronunciation for the source text.
  --aaw, --add_audio_words            Flag to add or update audio
                                      pronunciation for the extracted words
                                      from the text.
  --ascb, --add_save_comments_button  Flag to add 'Save Comments' button in
                                      the HTML page of the table.
                                      (experimental)
```

### Run the script `vocab` ⭐

1. Run the script:
   ```terminal
   vocab -t 'cielo y sol' -l en -m gemini --ap --as --aaw
   ```

   **NOTES**:
   - `-t 'cielo y sol'`: the source text to be translated from Spanish to English (`-l en`)
   - `-m gemini`: the `Gemini 1.0 Pro` model is used for translation from Spanish to English
   - `--ap`: Part-of-Speech (POS) tags will be added for each segmented word from the source text
   - `--as`: Two example sentences will be generated for each segmented word from the source text
   - `--aaw`: Audio pronounciations will be generated for each segmented word from the source text

3. The script will create an `audio/` directory (if it doesn't already exist) and save the
   audio files there. The CSV file will include clickable links to these audio files.

![Terminal output when running the script vocab](./docs/images/terminal_output_sol_y_cielo.png)

### Example CSV Structure

The CSV file might have the following structure:

| Word (Lang A) | Pinyin | Translation (Lang B) | POS  | Audio Path                     |
|---------------|--------|-----------------------|------|--------------------------------|
| 新词          | xīn cí | New word              | noun | file:///path/to/audio/xinci.wav |
| 例子          | lì zi  | Example               | noun | file:///path/to/audio/lizi.wav  |

## Known Issues and Limitations

- **Chinese Text-to-Speech:** For Chinese text, `MeloTTS` may have difficulties
  with single-character words and low volume on some words.
  - Spanish TTS is good except for very small words like "y".
  - Sound files are saved as `.wav.`

## Contributing

[Contributions](https://github.com/raul23/Vocab-Augmentor/pulls) are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of the language models and MeloTTS used in this project.
- Inspired by the need to make language learning more efficient and effective.
