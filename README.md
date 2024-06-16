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

1. Clone the repository:
   ```terminal
   git clone https://github.com/raul23/Vocab-Augmentor.git
   ```
2. Navigate to the project directory:
   ```terminal
   cd Vocab-Augmentor
   ```
3. Install the required dependencies:
   ```terminal
   pip install jieba langcodes language_data pandas pypinyin sacremoses spacy transformers
   ```
   <!--- TODO: pip install -r requirements.txt --->

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

### Run the script `vocab.py`

1. Prepare your vocabulary list and input text.
2. Run the script:
   ```terminal
   vocab.py --vocab_list your_vocab_list.csv --input_text your_text.txt --model_name chosen_model
   ```
3. The script will create an `audio/` directory (if it doesn't already exist) and save the audio files there. The CSV file will include clickable links to these audio files.

## Known Issues and Limitations

- **Chinese Text-to-Speech:** For Chinese text, `MeloTTS` may have difficulties
  with single-character words and low volume on some words.
  - Spanish TTS is good except for very small words like "y".
  - Sound files are saved as `.wav.`

### Example CSV Structure

The CSV file might have the following structure:

| Word (Lang A) | Pinyin | Translation (Lang B) | POS  | Audio Path                     |
|---------------|--------|-----------------------|------|--------------------------------|
| 新词          | xīn cí | New word              | noun | file:///path/to/audio/xinci.wav |
| 例子          | lì zi  | Example               | noun | file:///path/to/audio/lizi.wav  |

## Contributing

[Contributions](https://github.com/raul23/Vocab-Augmentor/pulls) are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of the language models and MeloTTS used in this project.
- Inspired by the need to make language learning more efficient and effective.
