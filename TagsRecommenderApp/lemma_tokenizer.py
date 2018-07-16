import pandas as pd
import numpy as np
from nltk import regexp_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class LemmaTokenizer(object):
    """
    tokenize text
    """
    def __init__(self, params):
        self.wnl = WordNetLemmatizer()
        self.stopwords = stopwords
        self.regexp_tokenize = regexp_tokenize

    def __call__(self, doc):
        if pd.notnull(doc):
            # add words to stoplist, previously punctuations have been removed,
            # so we should do the same for the stoplist
            # we also add the top 10 words in the stoplist, these top 10 words
            # are found after post-processing

            doc_lower = self.lower(doc)
            doc_punct = self.striphtmlpunct(doc_lower)
            doc_tabs = self.striptabs(doc_punct)

            # create stoplist
            stoplist = [self.striphtmlpunct(x)
                        for x in self.stopwords.words('english')] + [
                            'im', 'ive'] + [
                            'use', 'get', 'like', 'file', 'would', 'way',
                            'code','work', 'want', 'need']

            lemmatized = []
            regex_tokens = self.regexp_tokenize(doc_tabs,
                                                pattern='\w+\S+|\.\w+')

            for word in regex_tokens:
                #for word, p_tags in pos_tag(regex_tokens):
                #convert_pos_tag = convert_tag(p_tags)
                lemmatized_word = self.wnl.lemmatize(word)
                if lemmatized_word not in set(stoplist):
                    lemmatized.append(lemmatized_word)

            return lemmatized

        return pd.Series(doc)

    def striphtmlpunct(self, data):
        # remove html tags, code unnecessary punctuations
        # <.*?> to remove everything between <>
        # [^\w\s+\.\-\#\+] remove punctuations except .-#+
        # (\.{1,3})(?!\S) negative lookahead assertion: only match .{1,3} that
        # is followed by white space
        if pd.notnull(data):
            p = re.compile(r'<.*?>|[^\w\s+\.\-\#\+]')
            res = p.sub('', data)
            pe = re.compile('(\.{1,3})(?!\S)')

            return pe.sub('', res)
        return data

    def striptabs(self, data):
        # remove tabs breaklines
        p = re.compile(r'(\r\n)+|\r+|\n+|\t+/i')
        return p.sub(' ', data)

    def lower(self, data):
        if pd.notnull(data):
            return data.lower()
        return data
