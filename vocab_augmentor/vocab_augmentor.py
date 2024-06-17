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
