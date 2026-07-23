# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from .num import num2str
from cn2an import an2cn

# 温度表达式，温度会影响负号的读法
# -3°C 零下三度
RE_TEMPERATURE = re.compile(r"(-?)(\d+(\.\d+)?)(°C|℃)")
RE_TEMPERATURE_F = re.compile(r"(-?)(\d+(\.\d+)?)(°F|℉)")
measure_dict = {
    "cm2": "平方厘米",
    "cm²": "平方厘米",
    "cm3": "立方厘米",
    "cm³": "立方厘米",
    "cm": "厘米",
    "db": "分贝",
    "ds": "毫秒",
    "kg": "千克",
    "KG": "千克",
    "g": "克",
    "G": "克",
    "t": "吨",
    "T": "吨",
    "km": "千米",
    "KM": "千米",
    "m2": "平方米",
    "m²": "平方米",
    "m³": "立方米",
    "m3": "立方米",
    "ml": "毫升",
    "ML": "毫升",
    "m": "米",
    "mm": "毫米",
    "s": "秒",
    "h": "小时",
    "H": "小时",
    "l": "升",
    "L": "升",
    "bps": "比特每秒",
    "Bps": "字节每秒",
    "Pa": "帕斯卡",
    "J": "焦耳",
    "W": "瓦特",
}

currency_map = {
    "¥": "人民币",
    "$": "美元",
    "€": "欧元",
    "£": "英镑",
    "": "元"
}


def replace_temperature(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    temperature = match.group(2)
    unit = match.group(3)
    sign: str = "零下" if sign else ""
    temperature: str = num2str(temperature)
    #unit: str = "摄氏度" if unit in ["°C", "℃"] else "度"
    unit: str = "摄氏度"
    result = f"{sign}{temperature}{unit}"
    return result

def replace_temperature_f(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    temperature = match.group(2)
    unit = match.group(3)
    sign: str = "零下" if sign else ""
    temperature: str = num2str(temperature)
    #unit: str = "华氏度" if unit in ["°F", "℉"] else "度"
    unit: str = "华氏度"
    result = f"{sign}{temperature}{unit}"
    return result


def replace_measure(sentence) -> str:
    for q_notation in measure_dict:
        if q_notation in sentence:
            #pattern = re.compile(rf"(?<=\d){q_notation}(?=[^A-Za-z]|$)")
            #sentence = sentence.replace(q_notation, measure_dict[q_notation])
            #sentence = pattern.sub(measure_dict[q_notation], sentence)
            pattern = re.compile(
            rf'(?P<num>[+-]?\d[\d,]*(?:\.\d+)?)(?:\s*){re.escape(q_notation)}(?=[^A-Za-z]|$)'
            )

            def _repl(m: re.Match) -> str:
               num_str = m.group('num').replace(',', '')  # 去掉千分位逗号
               try:
                  num_cn = an2cn(num_str, "low")  # 100 -> 一百, 100.5 -> 一百点五
               except Exception:
                  num_cn = num_str  # 兜底
               return f'{num_cn}{measure_dict[q_notation]}'

            sentence = pattern.sub(_repl, sentence)
    return sentence

def replace_mixed_measure(sentence) -> str:
    pattern1 = re.compile(r"(¥|\$|€|£)?(\d+(?:\.\d+)?)(块|元)?/([A-Za-z]+)")

    def price_replacement(match):
        currency = match.group(1) or "" 
        amount = match.group(2)
        amount = an2cn(amount, "low")
        cn_money_word = match.group(3) or "" 
        unit = match.group(4)

        if currency in currency_map:
            currency_cn = currency_map[currency]
        elif cn_money_word in ["块", "元"]:
            currency_cn = "元"
        else:
            currency_cn = "元"

        unit_cn = measure_dict.get(unit, unit)
        return f"{amount}{currency_cn}每{unit_cn}"

    sentence = pattern1.sub(price_replacement, sentence)

    pattern2 = re.compile(r"(\d+(?:\.\d+)?(?:e\d+)?)?([A-Za-z]+[23²³]?)\/([A-Za-z]+[23²³]?)")

    def rules(match):
        number = match.group(1) or ""
        number = an2cn(number, "low")
        left = match.group(2)
        right = match.group(3)
        left_val = measure_dict.get(left, left)
        right_val = measure_dict.get(right, right)
        return f"{number}{left_val}每{right_val}"

    sentence =  pattern2.sub(rules, sentence)
    return sentence