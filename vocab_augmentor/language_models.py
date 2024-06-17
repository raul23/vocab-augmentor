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
