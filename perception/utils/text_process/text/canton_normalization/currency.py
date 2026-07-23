import re

def replace_currency_patterns(text):
    # Currency symbol mapping
    currency_map = {
        "$": "美元",
        "¥": "人民幣",
        "￥": "人民幣",
        "€": "歐元",
        "£": "英鎊"
    }
    
    # Regex: Match currency symbol followed by number
    pattern = re.compile(r"([￥¥$€£])\s*([\d,.]+)")
    
    # Replace using lambda
    return pattern.sub(lambda m: f"{m.group(2).replace(',', '')}{currency_map.get(m.group(1), '')}", text)

# Test
#data = ["$100", "¥200", "$1,500", "€3.5"]
#result = [convert_currency_string(x) for x in data]
#print(result)
