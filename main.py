import telebot
import onnxruntime
import sentencepiece as spm
import numpy as np
import pickle
import re

from navec import Navec
from slovnet import NER
from ipymarkup import show_span_ascii_markup as show_markup
from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)

flag = 0
bot = telebot.TeleBot('5803770980:AAFZEfCiFoZd3BVTKlqNftNDql7zQuQbWlM')
navec = Navec.load(r"/Users/ypxd/Desktop/DEPPLEARNING/dz2/models/navec_news_v1_1B_250K_300d_100q.tar")
ner = NER.load(r"/Users/ypxd/Desktop/DEPPLEARNING/dz2/models/slovnet_ner_news_v1.tar")
ner.navec(navec)

def get_myner(message, ner):
    response = " "
    text = message.text
    markup = ner(text)
    show_markup(markup.text, markup.spans)
    for span in markup.spans:
        print(text[span.start:span.stop])
        response = response + text[span.start:span.stop] + "\n"
    bot.send_message(message.from_user.id, response)

def get_normal_myner(message):
    response = ""
    text = message.text
    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    names_extractor = NamesExtractor(morph_vocab)

    doc = Doc(text)

    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.sents[0].morph.print()
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    doc.parse_syntax(syntax_parser)
    doc.sents[0].syntax.print()

    doc.tag_ner(ner_tagger)
    doc.ner.print()

    for span in doc.spans:
        span.normalize(morph_vocab)

    for span in doc.spans:
        if span.type == PER:
            span.extract_fact(names_extractor)
            response = response + "Origin text: %s, normalized: %s, first name: %s, last name: %s.\n" % (
                        span.text,
                        span.normal,
                        span.fact.as_dict['first'],
                        span.fact.as_dict['last']
            )
        else:
            response = response + "Origin text: %s.\n" % (
                span.text,
            )

    bot.send_message(message.from_user.id, response)


@bot.message_handler(content_types=['text'])
def get_telegram_ner(message):
    global flag
    if flag == 0:
        if message.text == "/start":
            bot.send_message(message.from_user.id,
                             "Приветствую! Данный бот предназначен для распознаванию имен людей, названий "
                             "организаций, топонимов. Для справки введи команду: /help")
        elif message.text == "/help":
            bot.send_message(message.from_user.id,
                             "Для перехода в режим генерации текста введи следующую команду: /ner")
        elif message.text == "/ner":
            bot.send_message(message.from_user.id, "Напиши текст, в котором нужно произвести поиск")
            flag = 1
        elif message.text == "/normal_ner":
            bot.send_message(message.from_user.id, "Напиши текст, в котором нужно произвести поиск")
            flag = 2
        else:
            bot.send_message(message.from_user.id, "Не понимаю, что тебе нужно. Для справки введи команду: /help")
    elif flag == 1:
        get_myner(message, ner)
        flag = 0

    elif flag == 2:
        get_normal_myner(message)
        flag = 0


bot.polling(none_stop=True, interval=0)