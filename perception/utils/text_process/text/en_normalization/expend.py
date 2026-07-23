# by https://github.com/Cosmo-klara

from __future__ import print_function

import re
import inflect
import unicodedata

from collections import OrderedDict
from .phrases import replace_phrases

# 后缀计量单位替换表
measurement_map = {
    "m": ["meter", "meters"],
    "cm":["centimeter", "centimeters"],
    "km": ["kilometer", "kilometers"],
    "km/h": ["kilometer per hour", "kilometers per hour"],
    "ft": ["feet", "feet"],
    "L": ["liter", "liters"],
    "tbsp": ["tablespoon", "tablespoons"],
    "tsp": ["teaspoon", "teaspoons"],
    "h": ["hour", "hours"],
    "min": ["minute", "minutes"],
    "s": ["second", "seconds"],
    "°C": ["degree celsius", "degrees celsius"],
    "°F": ["degree fahrenheit", "degrees fahrenheit"],
    "kg":['kilogram','kilograms'],
    "g":['gram','grams'],
    "t": ['ton','tons'],
    "db":['decibel','decibels'],
    "oz":['ounce','ounces'],
    "pb":['pound', 'pounds'],
    "in": ['inch', 'inches'],
    "cm2":["square centimeter", "square centimeters"],
    "cm²":["square centimeter", "square centimeters"],
    "m²":["square meter", "square meters"],
    "m2":["square meter", "square meters"],
    "cm3":["cube centimeter", "cube centimeters"],
    "cm³":["cube centimeter", "cube centimeters"],
    "m³":["cube meter", "cube meters"],
    "m3":["cube meter", "cube meters"],
    "sqft":["square feet", "square feet"],
    "ml":["milliliter", "milliliters"],
    "pa":["pascal", "pascal"],
    "W": ["watt", "watts"]
}


# 识别 12,000 类型
_inflect = inflect.engine()

# 转化数字序数词
_ordinal_number_re = re.compile(r"\b([0-9]+)\. ")

# 我听说好像对于数字正则识别其实用 \d 会好一点

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")

# 时间识别
_time_re = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")

# 后缀计量单位识别
_measurement_re = re.compile(r"\b([0-9]+(\.[0-9]+)?\s*(m|cm|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F|kg|t|db|oz|pb|in|cm2|cm²|cm3|cm³|m2|m3|m²|m³|sqft|ml|pa|W))\b")

# 前后 £ 识别 ( 写了识别两边某一边的，但是不知道为什么失败了┭┮﹏┭┮ )
_pounds_re_start = re.compile(r"£([0-9\.\,]*[0-9]+)")
_pounds_re_end = re.compile(r"([0-9\.\,]*[0-9]+)£")

# 前后 $ 识别
_dollars_re_start = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_dollars_re_end = re.compile(r"([(0-9\.\,]*[0-9]+)\$")

# 小数的识别
_decimal_number_re = re.compile(r"(-?)([0-9]+\.\s*[0-9]+)")
#_decimal_number_re = re.compile(r"(-?)((\d+)(\.\d+))" r"|(\.(\d+))")

# 分数识别 (形式 "3/4" )
_fraction_re = re.compile(r"(-?)([0-9]+/[0-9]+)")
#_fraction_re = re.compile(r"(-?)(\d+)/(\d+)")

# 序数词识别
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")

# 数字处理
_number_re = re.compile(r"([-+]?)([0-9]+)")


RE_NUMBER = re.compile(r"(-?)((\d+)(\.\d+)?)" r"|(\.(\d+))")

# 范围表达式
# match.group(1) and match.group(8) are copy from RE_NUMBER

RE_RANGE = re.compile(
    r"""
    (?<![\d\+\-\×\÷\=])      # 使用反向前瞻以确保数字范围之前没有其他数字和操作符
    ((-?)((\d+)(\.\d+)?))  # 匹配范围起始的负数或正数（整数或小数）
    \s*[-~]\s*                  # 匹配范围分隔符
    ((-?)((\d+)(\.\d+)?))\s* # 匹配范围结束的负数或正数（整数或小数）
    (?![\d\+\-\×\÷\=])       # 使用正向前瞻以确保数字范围之后没有其他数字和操作符
    """,
    re.VERBOSE,
)

NUM = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'  # 整数/小数，支持正负
UNIT = r'(?i:%|m|cm|km|km/h|kg/L|m/s|ft|L|tbsp|tsp|h|min|s|°C|°F|kg|t|db|oz|pb|in|cm2|cm²|cm3|cm³|m2|m²|m³|m3|sqft|ml|pa|W)'

RE_TO_RANGE = re.compile(
    rf'({NUM})\s*{UNIT}\s*(~|-)\s*({NUM})\s*{UNIT}\s*(?!=)'
)

# 次方专项
RE_POWER = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]+")

power_map = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "ˣ": "x",
    "ʸ": "y",
    "ⁿ": "n",
}

RE_SIMPLE_MINUS = re.compile(r"\b(\d+)\s*-\s*(\d+)\b")
RE_SIMPLE_RATIO = re.compile(r"(?<=\d):(?=\d)")
DIGITS = {str(i): tran for i, tran in enumerate(['zero','one','two','three','four','five','six','seven','eight','nine'])}

RE_ASMD = re.compile(
    r"((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))\s*([\+\-\×÷=])\s*((-?)((\d+)(\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|(\.\d+[⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*)|([A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ]*))"
)

asmd_map = {"+": " plus ", "-": " minus ", "×": " times ", "÷": " divided by ", "=": " equal to "}

RE_MOBILE_PHONE = re.compile(r"(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)")
RE_TELEPHONE = re.compile(r"(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)")
#RE_US_PHONE = re.compile(
    #r"(?<!\d)(?:\+?1[-.\s]?|1[-.\s]?|)?(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})(?!\d)"
#)


RE_US_PHONE = re.compile(
    r"""(?<!\d)                # 前缀不能是数字
    (?:\+?1[-.\s]?)?           # 可选的国家区号 +1
    \(?\d{3}\)?                # 区号，支持 (xxx) 或 xxx
    [-.\s]?                    # 可选分隔符
    \d{3}                      # 中间三位
    [-.\s]?                    # 可选分隔符
    \d{4}                      # 后四位
    (?!\d)                     # 后缀不能是数字
    """,
    re.VERBOSE
)

RE_NATIONAL_UNIFORM_NUMBER = re.compile(r"(400)(-)?\d{3}(-)?\d{4}")
RE_DATE2 = re.compile(r"(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])")



UNITS = OrderedDict(
    {
        1: "",
        2: "百",
        3: "千",
        4: "万",
        8: "亿",
    }
)




def _convert_ordinal(m):
    """
    标准化序数词, 例如: 1. 2. 3. 4. 5. 6.
    Examples:
        input: "1. "
        output: "1st"
    然后在后面的 _expand_ordinal, 将其转化为 first 这类的
    """
    ordinal = _inflect.ordinal(m.group(1))
    return ordinal + ", "


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_time(m):
    """
    将 24 小时制的时间转换为 12 小时制的时间表示方式。

    Examples:
        input: "13:00 / 4:00 / 13:30"
        output: "one o'clock p.m. / four o'clock am. / one thirty p.m."
    """
    hours, minutes = map(int, m.group(1, 2))
    period = "a.m." if hours < 12 else "p.m."
    if hours > 12:
        hours -= 12

    hour_word = _inflect.number_to_words(hours)
    #minute_word = _inflect.number_to_words(minutes) if minutes != 0 else ""
    minute_word = minutes if minutes != 0 else ""

    if minutes == 0:
        return f"{hour_word} o'clock {period}"
    else:
        return f"{hour_word} {minute_word} {period}"


def _expand_measurement(m):
    """
    处理一些常见的测量单位后缀, 目前支持: m, km, km/h, ft, L, tbsp, tsp, h, min, s, °C, °F
    如果要拓展的话修改: _measurement_re 和 measurement_map
    """
    sign = m.group(3)
    ptr = 1
    # 想不到怎么方便的取数字，又懒得改正则，诶，1.2 反正也是复数读法，干脆直接去掉 "."
    num = int(m.group(1).replace(sign, "").replace(".", ""))
    decimal_part = m.group(2)
    # 上面判断的漏洞，比如 0.1 的情况，在这里排除了
    if decimal_part == None and num == 1:
        ptr = 0
    return m.group(1).replace(sign, " " + measurement_map[sign][ptr])


def _expand_pounds(m):
    """
    没找到特别规范的说明，和美元的处理一样，其实可以把两个合并在一起
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " pounds"  # Unexpected format
    pounds = int(parts[0]) if parts[0] else 0
    pence = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if pounds and pence:
        pound_unit = "pound" if pounds == 1 else "pounds"
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s and %s %s" % (pounds, pound_unit, pence, penny_unit)
    elif pounds:
        pound_unit = "pound" if pounds == 1 else "pounds"
        return "%s %s" % (pounds, pound_unit)
    elif pence:
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s" % (pence, penny_unit)
    else:
        return "zero pounds"


def _expand_dollars(m):
    """
    change: 美分是 100 的限值, 应该要做补零的吧
    Example:
        input: "32.3$ / $6.24"
        output: "thirty-two dollars and thirty cents" / "six dollars and twenty-four cents"
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s and %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


# 小数的处理
def _expand_decimal_number(m):
    """
    Example:
        input: "13.234"
        output: "thirteen point two three four"
    """
    sign = m.group(1)
    if sign == "+":
        sign = "positive "
    elif sign == "-":
        sign = "negative "
    else:
        sign = ""
    match = m.group(2)
    parts = match.split(".")
    words = []
    # 遍历字符串中的每个字符
    for char in parts[1]:
        if char == ".":
            words.append("point")
        else:
            words.append(char)
    return sign + parts[0] + " point " + " ".join(words)


# 分数的处理
def _expend_fraction(m):
    """
    规则1: 分子使用基数词读法, 分母用序数词读法.
    规则2: 如果分子大于 1, 在读分母的时候使用序数词复数读法.
    规则3: 当分母为2的时候, 分母读做 half, 并且当分子大于 1 的时候, half 也要用复数读法, 读为 halves.
    Examples:

    | Written |	Said |
    |:---:|:---:|
    | 1/3 | one third |
    | 3/4 | three fourths |
    | 5/6 | five sixths |
    | 1/2 | one half |
    | 3/2 | three halves |
    """
    sign = m.group(1)
    if sign == "+":
        sign = "positive"
    elif sign == "-":
        sign = "negative"
    else:
        sign = ""
    match = m.group(2)
    numerator, denominator = map(int, match.split("/"))

    numerator_part = _inflect.number_to_words(numerator)
    if denominator == 2:
        if numerator == 1:
            denominator_part = "half"
        else:
            denominator_part = "halves"
    elif denominator == 1:
        return f"{numerator_part}"
    else:
        denominator_part = _inflect.ordinal(_inflect.number_to_words(denominator))
        if numerator > 1:
            denominator_part += "s"

    return f"{sign} {numerator_part} {denominator_part}"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    sign = m.group(1)
    #sign: str = "negative " if sign else ""
    if sign == "+":
        sign = "positive "
    elif sign == "-":
        sign = "negative "
    else:
        sign = ""
    num = int(m.group(2))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return sign + _inflect.number_to_words(num, andword="")

def replace_power(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    power_num = ""
    for m in match.group(0):
        power_num += power_map[m]
    result = " to the power of " + power_num
    return result

def replace_to_range(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    result = match.group(0).replace("~", " to ").replace("-", " to ")

    return result


def replace_asmd(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    result = match.group(1) + asmd_map[match.group(8)] + match.group(9)
    return result

def _get_value(value_string: str, use_zero: bool = True):
    stripped = value_string.lstrip("0")
    if len(stripped) == 0:
        return []
    elif len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS["0"], DIGITS[stripped]]
        else:
            return [DIGITS[stripped]]
    else:
        largest_unit = next(power for power in reversed(UNITS.keys()) if power < len(stripped))
        first_part = value_string[:-largest_unit]
        second_part = value_string[-largest_unit:]
        return _get_value(first_part) + [UNITS[largest_unit]] + _get_value(second_part)

def verbalize_cardinal(value_string: str) -> str:
    if not value_string:
        return ""

    # 000 -> '零' , 0 -> '零'
    value_string = value_string.lstrip("0")
    if len(value_string) == 0:
        return DIGITS["0"]

    result_symbols = _get_value(value_string)
    # verbalized number starting with '一十*' is abbreviated as `十*`
    if len(result_symbols) >= 2 and result_symbols[0] == DIGITS["1"] and result_symbols[1] == UNITS[1]:
        result_symbols = result_symbols[1:]
    return "".join(result_symbols)

def num2str(value_string: str) -> str:
    integer_decimal = value_string.split(".")
    if len(integer_decimal) == 1:
        integer = integer_decimal[0]
        decimal = ""
    elif len(integer_decimal) == 2:
        integer, decimal = integer_decimal
    else:
        raise ValueError(f"The value string: '${value_string}' has more than one point in it.")

    result = verbalize_cardinal(integer)

    if decimal.endswith("0"):
        decimal = decimal.rstrip("0") + "0"
    else:
        decimal = decimal.rstrip("0")

    if decimal:
        # '.22' is verbalized as '零点二二'
        # '3.20' is verbalized as '三点二
        result = result if result else "zero"
        result += "point" + verbalize_digit(decimal)
    return result

def replace_number(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    number = match.group(2)
    pure_decimal = match.group(5)
    if pure_decimal:
        result = num2str(pure_decimal)
    else:
        sign: str = "negative " if sign else ""
        number: str = num2str(number)
        result = f"{sign}{number}"
    return result

def replace_range(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    first, second = match.group(1), match.group(6)
    #first = RE_NUMBER.sub(replace_number, first)
    #second = RE_NUMBER.sub(replace_number, second)
    result = f"{first} to {second}"
    return result

def replace_mixed_measure(sentence) -> str:

    pattern = re.compile(r"(\d+(?:\.\d+)?(?:e\d+)?)?([A-Za-z]+[23²³]?)\/([A-Za-z]+[23²³]?)")

    def rules(match):
        number = match.group(1) or ""
        #number = RE_NUMBER.sub(replace_number, number)
        left = match.group(2)
        right = match.group(3)
        left_val = measurement_map.get(left, left)[0]
        right_val = measurement_map.get(right, right)[0]
        return f"{number} {left_val} per {right_val}"

    sentence =  pattern.sub(rules, sentence)
    return sentence


def verbalize_digit(value_string: str) -> str:
    result_symbols = [DIGITS[digit] for digit in value_string]
    result = " ".join(result_symbols)
    return result

def phone2str(phone_string: str, mobile=True) -> str:
    if mobile:
        sp_parts = phone_string.strip("+").split()
        result = "，".join([verbalize_digit(part) for part in sp_parts])
        return result
    else:
        sil_parts = phone_string.split("-")
        result = "，".join([verbalize_digit(part) for part in sil_parts])
        return result

def phone2str_us(phone_string: str) -> str:
    sp_parts = phone_string.strip("+").split()
    result = "，".join([verbalize_digit(part) for part in sp_parts])
    return result


def replace_usphone(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    return phone2str_us(digits)
    #return phone2str_us(match.group(0))


def replace_phone(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    return phone2str(match.group(0))


def replace_mobile(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    return phone2str(match.group(0))

def replace_date2(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    year = match.group(1)
    month = match.group(3)
    day = match.group(4)
    result = ""

    p = inflect.engine()
    months_en = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    if year:
        year_str = f"{p.number_to_words(int(year[:2])).replace('-',' ')} {p.number_to_words(int(year[2:])).replace('-',' ')}"
        #print(year_str)
    if month:
        month_str = months_en[int(month)]
    if day:
         day_str = p.number_to_words(day).replace('-',' ')

    result = f"{month_str}, {day_str}, {year_str}"
    return result



def normalize(text):
    """
    !!! 所有的处理都需要正确的输入 !!!
    可以添加新的处理，只需要添加正则表达式和对应的处理函数即可
    """
    text = replace_phrases(text)
    text = RE_DATE2.sub(replace_date2, text)

    text = RE_US_PHONE.sub(replace_usphone, text)
    text = RE_MOBILE_PHONE.sub(replace_mobile, text)

    text = RE_TO_RANGE.sub(replace_to_range, text)
    text = replace_mixed_measure(text)
    text = RE_RANGE.sub(replace_range, text)


    text = re.sub(_ordinal_number_re, _convert_ordinal, text)
    # text = re.sub(r"(?<!\d)-|-(?!\d)", " minus ", text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_pounds_re_start, _expand_pounds, text)
    text = re.sub(_pounds_re_end, _expand_pounds, text)
    text = re.sub(_dollars_re_start, _expand_dollars, text)
    text = re.sub(_dollars_re_end, _expand_dollars, text)

    text = re.sub("%", " percent ", text)
    text = re.sub("‰", " thousandth ", text)
    text = re.sub("‱", " ten thousandths", text)

    while RE_ASMD.search(text):
            text = RE_ASMD.sub(replace_asmd, text)
    text = RE_POWER.sub(replace_power, text)


    text = re.sub(_decimal_number_re, _expand_decimal_number, text)
    text = re.sub(_fraction_re, _expend_fraction, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    #text = RE_MOBILE_PHONE.sub(replace_mobile, text)
    #text = RE_US_PHONE.sub(replace_usphone, text)

    #text = RE_TELEPHONE.sub(replace_phone, text)

    #text = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, text)

    text = re.sub(_number_re, _expand_number, text)

    text = "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )  # Strip accents

    text = re.sub("&", " and ", text)
    text = re.sub("#", " hash", text)
    text = re.sub("/", " slash", text)
    text = re.sub("≈", " approximately equal to ", text)
    text = re.sub("≠", " not equal to ", text)
    text = re.sub("=", " equal to ", text)
    text = re.sub("\+", " plus ", text)
    text = re.sub("\×", " times ", text)
    text = re.sub("÷", " divided by ", text)
    text = re.sub(">", " large than ", text)
    text = re.sub("<", " less than ", text)
    text = re.sub("≥", " large or equal to  ", text)
    text = re.sub("≤", " less or equal to ", text)
    text = text.replace("@", " at")
    text = text.replace("∈‌", " belong to ")
    text = text.replace("√", " square root of ")
    text = text.replace("π", " pie ").replace("Π", " pie ")
    text = text.replace("≠", " not equal to ")
    text = text.replace("α", " alpha ")
    text = text.replace("β", " beta ")
    text = text.replace("γ", " gamma ").replace("Γ", " gamma ")
    text = text.replace("δ", " delta ").replace("Δ", " delta ")
    text = text.replace("ε", " epsilon ")
    text= text.replace("ζ", " zeta ")
    text = text.replace("η", " eta ")
    #text = RE_SIMPLE_MINUS.sub(r"\1 minus \2", text)
    # text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
    text = re.sub(r"(?i)i\.e\.", "that is", text)
    text = re.sub(r"(?i)e\.g\.", "for example", text)
    # 增加纯大写单词拆分
    text = re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)
    return text


if __name__ == "__main__":
    # 我觉得其实可以把切分结果展示出来（只读，或者修改不影响传给TTS的实际text）
    # 然后让用户确认后再输入给 TTS，可以让用户检查自己有没有不标准的输入
    print(normalize("1. test ordinal number 1st"))
    print(normalize("32.3$, $6.24, 1.1£, £7.14."))
    print(normalize("3/23, 1/2, 3/2, 1/3, 6/1"))
    print(normalize("1st, 22nd"))
    print(normalize("a test 20h, 1.2s, 1L, 0.1km"))
    print(normalize("a test of time 4:00, 13:00, 13:30"))
    print(normalize("a test of temperature 4°F, 23°C, -19°C"))
