"""
Translator Module

This module provides translation functionality using the IBM Watson Language Translator API.

"""

from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

#  IBM Watson Language Translator service
apikey = '<space>'
url = '<space>'

authenticator = IAMAuthenticator(apikey)
language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    authenticator=authenticator
)
language_translator.set_service_url(url)


def translate_text(source_language, target_language, text):
    """
    Translates the given text from the source language to the target language.

    Args:
        source_language (str): The source language code.
        target_language (str): The target language code.
        text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    translation = language_translator.translate(
        text=text,
        model_id=f'{source_language}-{target_language}'
    ).get_result()

    return translation['translations'][0]['translation']


def main():
    """
    Main function to interact with the translation functionality.
    """
    english_text = input("Enter English Text: ")
    translated = translate_text("en", "fr", english_text)
    print(translated)

    french_text = input("Enter French Text: ")
    translated = translate_text("fr", "en", french_text)
    print(translated)


if __name__ == "__main__":
    main()
