# Vocab-Augmentor [work-in-progress]
## Overview

**vocab.py** is a Python script designed to help language learners expand their vocabulary effortlessly. By leveraging advanced language models such as `facebook/mbart-large-50-many-to-many-mmt`, `Helsinki-NLP/opus-mt`, `Gemini 1.0 Pro`, and `llama 3-8b`, this tool identifies new words from any given text, translates them, and updates your personal vocabulary list. Additionally, it supports adding pronunciation sounds for new words using the MeloTTS text-to-speech library for supported languages.

## Features

- **Multi-Language Support**: Supports a variety of languages, making it versatile for learners of different languages.
- **Advanced Translation Models**: Choose from multiple state-of-the-art translation models for accurate translations.
- **Automatic Segmentation**: Segments the provided text to identify individual words.
- **New Vocabulary Detection**: Detects words not present in your existing vocabulary list.
- **Detailed Word Information**: Saves new words along with their translation, pinyin (if applicable), and part of speech (POS).
- **Audio Pronunciation**: Adds pronunciation sounds for new words using MeloTTS for supported languages, including English, French, Spanish, Chinese, Korean, and Japanese.
- **CSV Export**: Updates and exports the vocabulary list to a CSV file for easy access and further study, including clickable links to audio files.

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

## Usage

### Run the script

1. Prepare your vocabulary list and input text.
2. Run the script:
   ```terminal
   python vocab.py --vocab_list your_vocab_list.csv --input_text your_text.txt --model_name chosen_model
   ```
3. The script will create an `audio/` directory (if it doesn't already exist) and save the audio files there. The CSV file will include clickable links to these audio files.

### Example CSV Structure

The CSV file might have the following structure:

| Word (Lang A) | Pinyin | Translation (Lang B) | POS  | Audio Path                     |
|---------------|--------|-----------------------|------|--------------------------------|
| 新词          | xīn cí | New word              | noun | file:///path/to/audio/xinci.mp3 |
| 例子          | lì zi  | Example               | noun | file:///path/to/audio/lizi.mp3  |

## Contributing

[Contributions](https://github.com/raul23/Vocab-Augmentor/pulls) are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of the language models and MeloTTS used in this project.
- Inspired by the need to make language learning more efficient and effective.
