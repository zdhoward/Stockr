from pycoingecko import CoinGeckoAPI
from pprint import pprint

cg = CoinGeckoAPI()


def sanitize_number(num):
    num = round(num, 2)
    billions = 0
    millions = 0
    if num >= 1000000000:
        while num >= 1000000000:
            num -= 1000000000
            billions += 1
    elif num >= 1000000:
        while num >= 1000000:
            num -= 1000000
            millions += 1

    if billions > 0:
        num = f"{billions}B"
    elif millions > 0:
        num = f"{millions}M"

    return num


bitc_price = cg.get_price(
    ids="bitcoin,litecoin,ethereum",
    vs_currencies="cad",
    include_market_cap="true",
    include_24hr_vol="true",
    include_24hr_change="true",
)

# pprint(bitc_price)
# date format is dd-mm-yyyy
history = cg.get_coin_history_by_id(id="bitcoin", date="24-09-2020")

name = history["name"]
symbol = history["symbol"]

market_data = history["market_data"]

# pprint(market_data)

price = market_data["current_price"]["cad"]
market_cap = market_data["market_cap"]["cad"]
volume = market_data["total_volume"]["cad"]

price = sanitize_number(price)
market_cap = sanitize_number(market_cap)
volume = sanitize_number(volume)

print(f"{symbol}:\t{name}\t${price}\t${market_cap}\t{volume}")

# pprint(history)
