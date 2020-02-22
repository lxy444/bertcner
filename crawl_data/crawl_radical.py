# encoding=utf-8
''' code ref: https://github.com/WenDesi/Chinese_radical '''
''' 从百度字词上得到部首 '''

import pandas as pd
import urllib
from bs4 import BeautifulSoup
import csv
from urllib.parse import quote
import string
import os

class Radical(object):
    baiduhanyu_url = 'http://hanyu.baidu.com/zici/s?ptype=zici&wd=%s'

    def __init__(self, dictionary_filepath = 'radical_dict.txt'):
        self.dictionary_filepath = dictionary_filepath
        if os.path.exists(dictionary_filepath):
            print('load radical dictionary')
            self.read_dictionary()
        else:
            print('create radical dictionary')
            self.dictionary = dict()
        self.origin_len = len(self.dictionary)

    def read_dictionary(self):
        self.dictionary = dict()
        with open(self.dictionary_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                char, radical = line.strip().split(',')
                self.dictionary[char] = radical
        
    def write_dictionary(self):
        file_obj = open(self.dictionary_filepath, 'w', encoding='utf-8')
        for char, radical in self.dictionary.items():
            file_obj.write(char+','+radical+'\n')
        file_obj.close()

    def get_radical(self,word):
        if word in self.dictionary:
            return self.dictionary[word]
        else:
            return self.get_radical_from_baiduhanyu(word)

    def post_baidu(self,url):
        print(url)
        try:
            timeout = 5
            request = urllib.request.Request(url)
            #伪装HTTP请求
            request.add_header('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36')
            request.add_header('connection','keep-alive')
            request.add_header('referer', url)
            # request.add_header('Accept-Encoding', 'gzip')  # gzip可提高传输速率，但占用计算资源
            response = urllib.request.urlopen(request, timeout = timeout)
            html = response.read()
            response.close()
            return html
        except Exception as e:
            print('URL Request Error:', e)
            return None

    def anlysis_radical_from_html(self,html_doc):
        soup = BeautifulSoup(html_doc, 'html.parser')
        li = soup.find(id="radical")
        radical = li.span.contents[0]
        return radical

    def add_in_dictionary(self,word,radical):
        # add in file
        file_object = open(self.dictionary_filepath,'a+')
        file_object.write(word+','+radical+'\r\n')
        file_object.close()

        # refresh dictionary
        self.read_dictionary()

    def get_radical_from_baiduhanyu(self,word):
        url = self.baiduhanyu_url % word
        #-- edit: add following line to solve 'UnicodeEncodeError: 'ascii' codec can't encode characters'--#
        url = quote(url, safe=string.printable)  
        html = self.post_baidu(url)

        if html == None:
            return None

        radical = self.anlysis_radical_from_html(html)
        if radical != None:
            self.dictionary[word] = radical
        return radical

    def save(self):
        if len(self.dictionary) > self.origin_len:
            print('saved new dictionary')
            self.write_dictionary()

if __name__ == '__main__':
    r = Radical()
    print(r.get_radical('杻'))
    r.save()