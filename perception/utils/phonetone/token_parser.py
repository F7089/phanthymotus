"""WeText TokenParser without loading wetext's FST bundle (JP5 Python 3.8)."""

from __future__ import annotations

import string

EOS = "<EOS>"
TN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}
EN_TN_ORDERS = {
    "date": ["preserve_order", "text", "day", "month", "year"],
    "money": ["integer_part", "fractional_part", "quantity", "currency_maj"],
}
ITN_ORDERS = {
    "date": ["year", "month", "day", "preserve_order"],
    "fraction": ["sign", "numerator", "denominator"],
    "measure": ["numerator", "denominator", "value", "units"],
    "money": ["currency", "value", "decimal", "quantity"],
    "time": ["hour", "minute", "second", "noon", "zone"],
    "telephone": ["country_code", "number_part"],
    "electronic": ["username", "domain", "protocol"],
}
EN_ITN_ORDERS = ITN_ORDERS


def escape_value(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')


class Token:
    def __init__(self, name, start=None):
        self.name = name
        self.start = start
        self.end = None
        self.order = []
        self.members = {}

    def append(self, key, value):
        self.order.append(key)
        self.members[key] = value

    def string(self, orders):
        output = self.name + " {"
        order = self.order
        if self.name in orders and (
            "preserve_order" not in self.members or self.members["preserve_order"] != "true"
        ):
            canonical_order = orders[self.name]
            order = canonical_order + [key for key in self.order if key not in canonical_order]
        for key in order:
            if key not in self.members:
                continue
            output += ' {}: "{}"'.format(key, escape_value(self.members[key]))
        return output + " }"


class TokenParser:
    def __init__(self, lang, operator="tn"):
        if lang not in ("en", "zh", "ja"):
            raise ValueError(lang)
        if lang == "en":
            self.orders = EN_TN_ORDERS if operator == "tn" else EN_ITN_ORDERS
        else:
            self.orders = TN_ORDERS if operator == "tn" else ITN_ORDERS

    def load(self, input):
        self.index = 0
        self.text = input
        self.char = input[0]
        self.tokens = []

    def read(self):
        if self.index < len(self.text) - 1:
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False

    def parse_ws(self):
        not_eos = self.char != EOS
        while not_eos and self.char == " ":
            not_eos = self.read()
        return not_eos

    def parse_char(self, exp):
        if self.char == exp:
            self.read()
            return True
        return False

    def parse_chars(self, exp):
        ok = False
        for x in exp:
            ok |= self.parse_char(x)
        return ok

    def parse_key(self):
        key = ""
        while self.char in string.ascii_letters + "_":
            key += self.char
            self.read()
        return key

    def parse_value(self):
        escape = False
        value = ""
        while self.char != '"':
            value += self.char
            escape = self.char == "\\"
            self.read()
            if escape:
                escape = False
                value += self.char
                self.read()
        return value

    def parse(self, input):
        self.load(input)
        while self.parse_ws():
            name = self.parse_key()
            self.parse_chars(" { ")
            token = Token(name)
            while self.parse_ws():
                if self.char == "}":
                    self.parse_char("}")
                    break
                key = self.parse_key()
                self.parse_chars(': "')
                value = self.parse_value()
                self.parse_char('"')
                token.append(key, value)
            self.tokens.append(token)

    def reorder(self, input):
        self.parse(input)
        return " ".join(token.string(self.orders) for token in self.tokens)
