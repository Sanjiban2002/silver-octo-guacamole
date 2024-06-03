#!/usr/bin/env python

def has_title(soup):
    if soup.title is None:
        return 0
    if len(soup.title.text) > 0:
        return 1
    else:
        return 0


def has_input(soup):
    if len(soup.find_all("input")):
        return 1
    else:
        return 0


def has_button(soup):
    if len(soup.find_all("button")) > 0:
        return 1
    else:
        return 0


def has_image(soup):
    if len(soup.find_all("image")) == 0:
        return 0
    else:
        return 1


def has_submit(soup):
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
        else:
            pass
    return 0


def has_link(soup):
    if len(soup.find_all("link")) > 0:
        return 1
    else:
        return 0


def has_password_input(soup):
    for input in soup.find_all("input"):
        if (input.get("type") or input.get("name") or input.get("id")) == "password":
            return 1
        else:
            pass
    return 0


def has_email_input(soup):
    for input in soup.find_all("input"):
        if (input.get("type") or input.get("id") or input.get("name")) == "email":
            return 1
        else:
            pass
    return 0


def has_hidden_element_input(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "hidden":
            return 1
        else:
            pass
    return 0


def has_audio(soup):
    if len(soup.find_all("audio")) > 0:
        return 1
    else:
        return 0


def has_video(soup):
    if len(soup.find_all("video")) > 0:
        return 1
    else:
        return 0


def has_h1(soup):
    if len(soup.find_all("h1")) > 0:
        return 1
    else:
        return 0


def has_h2(soup):
    if len(soup.find_all("h2")) > 0:
        return 1
    else:
        return 0


def has_h3(soup):
    if len(soup.find_all("h3")) > 0:
        return 1
    else:
        return 0


def has_footer(soup):
    if len(soup.find_all("footer")) > 0:
        return 1
    else:
        return 0


def has_form(soup):
    if len(soup.find_all("form")) > 0:
        return 1
    else:
        return 0


def has_textarea(soup):
    if len(soup.find_all("textarea")) > 0:
        return 1
    else:
        return 0


def has_iframe(soup):
    if len(soup.find_all("iframe")) > 0:
        return 1
    else:
        return 0


def has_text_input(soup):
    for input in soup.find_all("input"):
        if input.get("type") == "text":
            return 1
    return 0


def has_nav(soup):
    if len(soup.find_all("nav")) > 0:
        return 1
    else:
        return 0


def has_object(soup):
    if len(soup.find_all("object")) > 0:
        return 1
    else:
        return 0


def has_picture(soup):
    if len(soup.find_all("picture")) > 0:
        return 1
    else:
        return 0


def number_of_input(soup):
    return len(soup.find_all("input"))


def number_of_button(soup):
    return len(soup.find_all("button"))


def number_of_image(soup):
    image_tags = len(soup.find_all("image"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") or meta.get("name") == "image":
            count += 1
    return image_tags + count


def number_of_option(soup):
    return len(soup.find_all("option"))


def number_of_list(soup):
    return len(soup.find_all("li"))


def number_of_table_header(soup):
    return len(soup.find_all("th"))


def number_of_table_row(soup):
    return len(soup.find_all("tr"))


def number_of_hyperlink(soup):
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count


def number_of_paragraph(soup):
    return len(soup.find_all("p"))


def number_of_script(soup):
    return len(soup.find_all("script"))


def length_of_title(soup):
    if soup.title == None:
        return 0
    return len(soup.title.text)


def length_of_text(soup):
    return len(soup.get_text())


def number_of_clickable_button(soup):
    count = 0
    for button in soup.find_all("button"):
        if button.get("type") == "button":
            count += 1
    return count


def number_of_a(soup):
    return len(soup.find_all("a"))


def number_of_img(soup):
    return len(soup.find_all("img"))


def number_of_div(soup):
    return len(soup.find_all("div"))


def number_of_figure(soup):
    return len(soup.find_all("figure"))


def number_of_meta(soup):
    return len(soup.find_all("meta"))


def number_of_source(soup):
    return len(soup.find_all("source"))


def number_of_span(soup):
    return len(soup.find_all("span"))


def number_of_table(soup):
    return len(soup.find_all("table"))
