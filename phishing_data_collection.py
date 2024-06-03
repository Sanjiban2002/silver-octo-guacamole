#!/usr/bin/env python

import requests as re
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe

disable_warnings(InsecureRequestWarning)

file_name = "verified_online.csv"
data_frame = pd.read_csv(file_name)

URL_list = data_frame['url'].to_list()

total_urls = len(URL_list)
chunk_size = 100

all_data = []


def create_structured_data(url_list):
    data_list = []
    for i in range(0, len(url_list)):
        try:
            response = re.get(url_list[i], verify=False, timeout=4)
            if response.status_code != 200:
                print(i, "Connection was not successful for the URL ", url_list[i])
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(url_list[i]))
                all_data.append(vector)
        except re.exceptions.RequestException as e:
            print(i, " --> ", e)
            continue


def process_urls_in_chunks():
    for begin in range(0, total_urls, chunk_size):
        end = min(begin + chunk_size, total_urls)
        collection_list = URL_list[begin:end]

        create_structured_data(collection_list)


process_urls_in_chunks()

columns = [
    'has_title',
    'has_input',
    'has_button',
    'has_image',
    'has_submit',
    'has_link',
    'has_password_input',
    'has_email_input',
    'has_hidden_element_input',
    'has_audio',
    'has_video',
    'has_h1',
    'has_h2',
    'has_h3',
    'has_footer',
    'has_form',
    'has_textarea',
    'has_iframe',
    'has_text_input',
    'has_nav',
    'has_object',
    'has_picture',
    'number_of_input',
    'number_of_button',
    'number_of_image',
    'number_of_option',
    'number_of_list',
    'number_of_table_header',
    'number_of_table_row',
    'number_of_hyperlink',
    'number_of_paragraph',
    'number_of_script',
    'length_of_title',
    'length_of_text',
    'number_of_clickable_button',
    'number_of_a',
    'number_of_img',
    'number_of_div',
    'number_of_figure',
    'number_of_meta',
    'number_of_source',
    'number_of_span',
    'number_of_table',
    'URL'
]

df = pd.DataFrame(data=all_data, columns=columns)

df['label'] = 1

df.to_csv("structured_data_phishing.csv", index=False)

print("Finished processing all URL chunks!")
