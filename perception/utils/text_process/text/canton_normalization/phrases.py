import re

mapping_dic = {
    "yyds": "永遠嘅神",
    "xswl": "笑死我了",
    "u1s1": "有一講一",
    "nbcs": "no body cares",
    "nsdd": "你講嘅對",
    "zqsg": "真情實感",
    "gkd": "搞快點",
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
    "ByeBye": "bye bye",
    "Wifi": "Y fi",
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

