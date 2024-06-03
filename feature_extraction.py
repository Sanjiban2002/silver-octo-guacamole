#!/usr/bin/env python

import importlib
fd = importlib.import_module("content-based_feature_definition")


def create_vector(soup):
    return [
        fd.has_title(soup),
        fd.has_input(soup),
        fd.has_button(soup),
        fd.has_image(soup),
        fd.has_submit(soup),
        fd.has_link(soup),
        fd.has_password_input(soup),
        fd.has_email_input(soup),
        fd.has_hidden_element_input(soup),
        fd.has_audio(soup),
        fd.has_video(soup),
        fd.has_h1(soup),
        fd.has_h2(soup),
        fd.has_h3(soup),
        fd.has_footer(soup),
        fd.has_form(soup),
        fd.has_textarea(soup),
        fd.has_iframe(soup),
        fd.has_text_input(soup),
        fd.has_nav(soup),
        fd.has_object(soup),
        fd.has_picture(soup),
        fd.number_of_input(soup),
        fd.number_of_button(soup),
        fd.number_of_image(soup),
        fd.number_of_option(soup),
        fd.number_of_list(soup),
        fd.number_of_table_header(soup),
        fd.number_of_table_row(soup),
        fd.number_of_hyperlink(soup),
        fd.number_of_paragraph(soup),
        fd.number_of_script(soup),
        fd.length_of_title(soup),
        fd.length_of_text(soup),
        fd.number_of_clickable_button(soup),
        fd.number_of_a(soup),
        fd.number_of_img(soup),
        fd.number_of_div(soup),
        fd.number_of_figure(soup),
        fd.number_of_meta(soup),
        fd.number_of_source(soup),
        fd.number_of_span(soup),
        fd.number_of_table(soup)
    ]
