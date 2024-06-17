# Vocab-Augmentor [work-in-progress]
## Overview

**vocab.py** is a Python script designed to help language learners expand their 
vocabulary effortlessly. By leveraging advanced language models such as
`facebook/mbart-large-50-many-to-many-mmt`, `Helsinki-NLP/opus-mt`,
`Gemini 1.0 Pro`, and `llama 3-8b`, this tool identifies new words from any
given text, translates them, and updates your personal vocabulary list.
Additionally, it supports adding pronunciation sounds for new words using the
`MeloTTS` and `facebook/mms-tts-por` text-to-speech libraries for supported
languages.

## Features

- **Multi-Language Support**: Supports a variety of languages, making it versatile for learners of different languages.
- **Advanced Translation Models**: Choose from multiple state-of-the-art translation models for accurate translations.
- **Automatic Segmentation**: Segments the provided text to identify individual words.
- **New Vocabulary Detection**: Detects words not present in your existing vocabulary list.
- **Detailed Word Information**: Saves new words along with their translation, pinyin (if applicable), and part of speech (POS).
- **Audio Pronunciation**: Adds pronunciation sounds for new words using MeloTTS for supported languages, including English, French, Spanish, Chinese, Korean, and Japanese.
- **CSV Export**: Updates and exports the vocabulary list to a CSV file for easy access and further study, including clickable links to audio files.

## Text Generation

- **Text Generation**: Use either `Gemini 1.0 Pro` or `llama 3-8b` to generate example sentences.
  - `Gemini 1.0 Pro` is faster as it uses an API.

## How It Works

1. **Input**: Provide a list of known vocabulary words from language A with translations in language B.
2. **Text Processing**: Input text in language A for processing.
3. **Segmentation & Translation**: The script segments the text, translates each word using the chosen language model, and checks if the word is new.
4. **Add Pronunciation**: For new words in supported languages, generate pronunciation sounds using the MeloTTS text-to-speech library and save the audio files in a designated directory (e.g., `audio/`).
5. **Update Vocabulary List**: Adds new words to the vocabulary list along with their translation, pinyin (if applicable), POS, and clickable links to the pronunciation sound files.
6. **Export**: The updated vocabulary list is saved to a CSV file, with clickable links to the audio files included.

## Installation

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

## Performance Recommendations

- **GPU Recommendation**: When using `llama 3-8b`, GPU usage is highly recommended 
  for faster processing.

## API Key Management

To utilize the advanced translation and text generation features of
`Gemini 1.0 Pro` and `llama 3-8b`, API keys are required. These keys must be
saved as environment variables or can be retrieved from Kaggle Secrets. Follow
the steps below to manage your API keys:

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

### Using Kaggle Secrets

If you prefer to use Kaggle Secrets to manage your API keys, the script will
automatically attempt to retrieve the keys if they are not found in the environment
variables.

1. **Store API Keys in Kaggle Secrets**
   - In your Kaggle notebook, navigate to the "Add-ons" tab, select "Secrets", 
     and add your keys:
     - Key Name: `GEMINI_API_KEY`
     - Key Value: `your_gemini_api_key`
     - Key Name: `HF_API_KEY`
     - Key Value: `your_hugging_face_api_key`

2. **Access API Keys in the Script**
   - The script includes logic to check for the keys in the Kaggle Secrets if 
     they are not found in the environment variables. No additional steps are 
     required.

### Important Notes

- Ensure your API keys are kept confidential and not shared publicly.
- The script prioritizes environment variables over Kaggle Secrets. If both are 
  set, the environment variables will be used.
- Using API keys allows the script to access powerful language models and 
  generate accurate translations and text examples efficiently.

By following these steps, you can seamlessly integrate API keys into the 
Vocab-Augmentor script and leverage its full capabilities for advanced language 
learning tasks.

## Usage

### Script options

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

### Run the script `vocab`

1. Run the script:
   ```terminal
   vocab -t 'cielo y sol' -l en -m gemini --aaw
   ```
2. The script will create an `audio/` directory (if it doesn't already exist) and save the audio files there. The CSV file will include clickable links to these audio files.

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
