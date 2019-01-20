# you can use something like this if read_html fails to find a table
# if you have bs4 >= 4.2.1, you can skip the lxml stuff, the tables
# are scraped automatically. 4.2.0 won't work.
#
import pandas as pd
from lxml import html
import numpy as np
word = ""
words = []
def english_extractor():
    df = pd.read_html('http://ro.talkenglish.com/vocabulary/top-2000-vocabulary.aspx')

    new_df = df[1]


    what_i_need = new_df[[1]]

    what_i_need = pd.DataFrame([what_i_need])


    #file = open("cuvinte.txt","r")

    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        what_i_need.to_csv(r'C:\Users\alex_\PycharmProjects\Licenta\cuvinte\cuvinte.txt', header=None, index=None, sep=' ', mode='a')

def data_cleaning():
    global words
    global word
    file = open(r"C:\Users\alex_\PycharmProjects\Licenta\cuvinte\cuvinte.txt","r")
    x = file.readlines()
    for line in x:
        for char in line[4:]:
            if char == " ":
                continue
            else:
                word = word + char
        words.append(word)
        word = ""
    return words
data_cleaning()
