# -*- coding: utf-8 -*-
"""
This file contains utilities to simplify Kraken investment wrapper script.

Created on Sat Jul 18 15:28:53 2020

@author: boris
"""

import krakenex

import decimal
import time
import datetime
import math

class Utilities:
    """
    Utilities is a class that contains different methods to simplify crypto_wrapper.py
    """
    def __init__(self):
        pass

    def is_error(self, query_output, currency):
        """
        Method that verifies if an error occurred during the interaction with Kraken
        platform, and either writes the error into a specific output path and print it
        to the stdout, or return a 0 if no error occurred.
        query_output : Dictionary output obtained from the interaction with Kraken platform
        currency : ID of the currency used in the current loop
        """
        if query_output["error"]:  # if error occurs while querying info on Kraken, write it
            with open("data/{}_records.txt".format(currency), "a") as balance_file:
                balance_file.write("{}\n".format(query_output["error"]))
            print(query_output["error"])
            time.sleep(10)
            return 1
        else:
            return 0

    def eval_fee(self, fee_list, base_volume):
        """
        Method that evaluates the fee that will be applied based on the current volume
        invested.
        fee_list : List of fee rates that will be applied depending on the volume
        invested
        base_volume : Volume of the base currency to be invested

        Returns the final fee that will be applied to the investment.
        """
        final_fee = 0.0
        for fee in fee_list:
            if base_volume >= float(fee[0]):
                final_fee = fee[1]
        final_fee /= 100  # Convert percent in rate
        return final_fee

    def decimal_round(self, decimal_nb):
        """
        Method that defines what is the ideal value to add to a float to ceil it
        relatively high to close an order.
        decimal_nb : Number of decimal of the value to be ceiled

        The output of this method is the a number to be summed to the value that needs
        to be ceiled.
        """
        if int(decimal_nb) >= 4:
            return round(5*10**-int(decimal_nb), decimal_nb)  # Note that the round function is used due to the addition of decimals by the 10**n function that is inaccurate in python
        else:
            return round(2*10**-int(decimal_nb), decimal_nb)  # add 2*10**decimal to value

    def float_to_str(self, f):
        """
        Method that converts the given float to a string, without resorting to scientific
        notation.
        """
        # create a new context for this task
        ctx = decimal.Context()
        # 20 digits should be enough for everyone :D
        ctx.prec = 20
        d1 = ctx.create_decimal(repr(f))
        return format(d1, 'f')

    def volume_round(self, volume_to_invest, order_price):
        """
        Method that rounds down the volume to be invested in function of the price
        of the currency to be ordered.
        volume_to_invest : Volume to invest in a currency
        order_price : Price per currency of the concerned currency

        The output of this method is the floored volume of currency to be ordered.
        """
        if float(order_price) < 5:
            return float(math.floor(volume_to_invest))  # Round value without decimal if crypto price is inferior to 5 per crypto
        else:  # round at second decimal after the first non-zero, e.g. 0.00156 -> 0.0015
            rounding_str = str(volume_to_invest)
            if "." in rounding_str:
                rounding_str = self.float_to_str(volume_to_invest)  # Avoid scientific notation
                unit, decimal = rounding_str.split(".")
                count_non_zero = 0  # Used as a counter of non-zero decimals + zero decimals once the first non-zero number is found
                total_decimal_count = 0  # total decimal count
                first_non_zero = False  # Becomes true once first non-zero decimal is encountered
                for char in decimal:  # count the number of 0 before the first number in the decimal and then accept two numbers
                    if count_non_zero == 2:  # Break once 2 decimals have been counted after the first non-zero value
                        break
                    elif first_non_zero:
                        count_non_zero += 1
                        total_decimal_count += 1
                    elif char != "0":
                        first_non_zero = True
                        count_non_zero += 1
                        total_decimal_count += 1
                    elif char == "0":
                        total_decimal_count += 1
                return float("{}.{}".format(unit, decimal[:total_decimal_count]))
            else:
                return volume_to_invest  # If no decimal, no need to round

    def is_min_vol(self, volume_to_invest, min_vol_order, currency, algorithm, benchmark):
        """
        Method that verifies if the volume to be invested reaches the minimum volume
        of a specific currency that can be bought / sold with Kraken platform.
        volume_to_invest : Volume of currency to be invested
        min_vol_order : Minimal volume that can be invested for the currency of interest
        currency : ID of the currency used in the current loop

        The method will return 0 if the minimum volume is not reached, and 1 if it
        is reached.
        """
        if volume_to_invest < float(min_vol_order):
            with open("{}_{}/{}_records.txt".format(algorithm, benchmark, currency), "a") as balance_file:
                balance_file.write("Not enough crypto volume invested\n")
            return 0
        else:
            return 1

    def is_zero_funds(self, funds, currency):
        """
        Method that verifies if any funds of the currency of interest are available.
        funds : Dictionary that contains the funds available per currency
        currency : ID of the currency used in the current loop

        The method will return the string '0' if no funds are available for the currency
        of interest, and a string of the amount of the currency available on the Kraken
        account if some funds of the currency are available.
        """
        if funds.get(currency):
            return funds[currency]
        else:
            return "0"

