import re

mapping_dic = {
    "ASAP": "as soon as possible",
    "FYI": "For your information",
    "BTW": "By the way",
    "OMG": "Oh my god",
    "THX": "Thanks",
    "BBQ": "Barbecue",
    "LOL": "Laugh out loud",
    "LMAO": "Laugh my ass off",
    "sin": "signing",
    "cos": "co signing",
    "tan": "tangent",
    "sec": "sec tangent",
    "cot": "co tangent",
    "csc": "co sec tangent",
    "C#": "C sharp",
    "Chatgpt": "chat gpt",
    "Wifi": "Y Fi",
    " 911": " Nine One One ",
    "to-do": "to do",
    "Si-Fi": "si fi"
}

def replace_phrases(text):

    for phrase, replacement in mapping_dic.items():
            if phrase in text:
                text = text.replace(phrase, replacement)
            elif phrase.upper() in text:
                text = text.replace(phrase.upper(), replacement)
            elif phrase.lower() in text:
                text = text.replace(phrase.lower(), replacement)


    return text

