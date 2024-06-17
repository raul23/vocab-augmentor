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
