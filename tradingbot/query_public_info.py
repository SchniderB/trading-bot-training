# -*- coding: utf-8 -*-
"""
This file contains utilities for accessing different information for the currencies
available on Kraken platform.

Created on Sat Jul 18 15:28:53 2020

@author: boris
"""

import krakenex

import decimal
import time
import datetime

class Query_public_info:
    """
    Query_public_info is a class that extracts information about currencies from
    kraken online trading platform through the use of krakenex API.
    """
    def __init__(self):
        self.k = krakenex.API()  # Load class from krakenex API

    def get_all_info_assetPairs(self, currency_pair = "", info = ""):
        """
        Method that extracts all the information available on each pair of assets.
        """
        data = dict()
        if currency_pair: data["pair"] = currency_pair
        if info: data["info"] = info
        return self.k.query_public('AssetPairs', data = data)


if __name__ == '__main__':
    get_info = Query_public_info()
    resp = get_info.get_all_info_assetPairs(currency_pair = "ALGOEUR")
    print(resp)
