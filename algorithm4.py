# -*- coding: utf-8 -*-
"""
This script performs the benchmark of the algorithm4 on all the cryptocurrencies
that passed the pre-evaluation.

Created on Sun Sep 27 15:28:53 2020

@author: boris
"""

#### Import built-in and third-party modules and packages ####
import krakenex
import decimal
import time
import datetime
import statistics
import json
import os
from itertools import product
# from multiprocessing import Pool
import concurrent.futures  # Alternative to multiprocessing library
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
####

#### Import home-made modules and packages ####
from tradingbot import *
####

#### Define functions that will be used for the benchmark by the main benchmark function ####
#### Function that extracts the first and last trade time for a specific currency ####
def extract_first_last_time(file_name):
    with open(file_name, "r") as f:
        f.readline()
        first_line = f.readline()
    with open(file_name, "rb") as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()

    return float(first_line.split("\t")[2]), float(last_line.split("\t")[2])
####

#### Function that computes linear regression ####
def compute_linear_regression(time, price):
    """
    Function that computes the slope, the mean squared error and the coefficient of
    determination of a linear regression fitted on the evolution of the price in function
    of the time.
    """
    time = np.array(time).reshape((-1, 1))
    price = np.array(price)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(time, price)

    # Return the results of the statistics
    return regr.coef_  # mean_squared_error(future_time, future_prices), r2_score(future_time, future_prices)
####
####

#### Define list of parameters to benchmark the algorithm ####
min_slope = [0, -0.1, -0.25, -0.5, -1, -2, -5]  # cut-off of the slope below which the crypto will be sold
sell_slope_history = [10, 60, 24*60, 7*24*60, 30*24*60]  # Size of the history to include in the slope computation
all_param_comb = [[*param, i] for i, param in enumerate(list(product(min_slope, sell_slope_history)))]
####

#### Instanciate classes ####
k = krakenex.API()
get_info = query_public_info.Query_public_info()
utilities = utilities.Utilities()
####

#### Define list of crypto to include in the benchmark ####
all_pairs = []
currencies = []
data = json.load( open( "../crypto_repertoire.json" ) )
crypto_IDs = data.keys()
list_files = [file for file in sorted(os.listdir("..")) if "close_price" in file]
for crypto in crypto_IDs:
    if data[crypto]["pre-evaluation"] == "PASSED":
        for file in list_files:
            if crypto in file:
                all_pairs.append(file.split("_")[0])
                currencies.append(crypto)
                break
base_currency = "ZEUR"
####

#### Define the constant metrics ####
constant_metrics = dict()  # "pair" = [decimals, min_vol_order, fees]
for i, pair in enumerate(all_pairs):
    time.sleep(0.5)
    gen_info = get_info.get_all_info_assetPairs(currency_pair = pair)
    if utilities.is_error(gen_info, currencies[i]):  # If error occurs while querying info, write it
        continue  # Cannot start the investment loop if an error occurred here
    else:
        constant_metrics[pair] = [gen_info["result"][pair]["pair_decimals"], gen_info["result"][pair]["ordermin"], gen_info["result"][pair]["fees"]]
####

#### Count full benchmark execution time ####
chrono_start = time.time()
####

#### Encapsulate benchmark into a whole function ####
def run_benchmark(param):
    """
    Function that performs the benchmark
    """

    #### Verify that the history size makes sense in terms of number of time steps ####
    time_step = 60*2  # Hardcoded variable that was tested with algo 1 and 2
    if time_step*param[1] > 4*30*24*3600:  # Should not happen with the current set of parameters
        return 0
    else:
        benchmark = "benchmark{}".format(param[2])
    ####
    print("Started benchmark number {}".format(param[2]))
    #### Define base variables and instanciate classes ####
    algorithm = "algorithm4"
    sum_to_invest = [0.9 for i in range(len(all_pairs))]  # Fraction of the total balance to invest per crypto
    price_last_decision = {currency:[] for currency in currencies}  # simple memory of the last price and volume, contains ["decision", closing, volume, fee]
    history = {currency:[] for currency in currencies}  # operation history
    funds = {currency:0 for currency in currencies}
    funds[base_currency] = [1000 for i in range(len(currencies))]  # Virtual 1000 EUR available per currency
    ####

    #### Create output directory specific to the algorithm and benchmark number, and write output file ####
    os.mkdir("{}_{}".format(algorithm, benchmark))
    for currency in currencies:
        with open("{}_{}/{}_records.txt".format(algorithm, benchmark, currency), "w") as balance_file:
            balance_file.write("round\ttime\toperation\torder_price\tdecimal\tfee\tmin_order_volume\tvolume_to_invest\t{0}_balance\t{1}_balance\t{1}_rate\tbenefit\ttime_stamp\n".format(base_currency, currency))
    ####

    #### Write parameter repertoire ####
    with open("{}_{}/pipeline_parameters.txt".format(algorithm, benchmark), "w") as param_file:
        param_file.write("minimal_slope_before_sell\tminimal_sell_history_size_regression\tbenchmark_number\n{}\t{}\t{}".format(*param))
    ####

    #### Start investment loop ####
    for i, currency in enumerate(currencies):

        # print("Started benchmark on {}".format(currency))
        #### Define variables that are reset at each new loop ####
        loop_nb = 0
        pair = all_pairs[i]  # Define currency pair
        min_sell_history = 21600  # Based on the results of algo3.2, 3.3, 3.4, 3.5
        slope = param[0]
        history_regression_sell = param[1]
        min_benef = 1.05  # Based on the results of algo3: valid only in combination with min_hist
        min_hist = 10  # Based on the results of algo3: valid only in combination with min_benef
        ####

        #### Full code into try to avoid crashes if Kraken platform does not respond ####
        try:
        ####

            #### Extract fee per purchase per currency pair + decimal limit + min volume order ####
            decimal = constant_metrics[pair][0]  # decimal of rouding allowed for crypto
            min_vol_order = constant_metrics[pair][1]  # minimal volume of crypto per order
            fees = constant_metrics[pair][2]  # fees applied depending on the volume ordered
            ####

            #### Extract first and last trade of currency ####
            time, end_time = extract_first_last_time("../{}_close_price.txt".format(pair))  # Read the price history files
            time += 3600*24*7  # starts only in 2020
            end_time -= 3600*24*7  # Define end date one week before the end
            ####

            #### Start while loop with time ####
            with open("../{}_close_price.txt".format(pair), "r") as close_price_file:  # Read the price history files
                close_price_file.readline()  # Skip header
                j = 0
                increment = True
                while time < end_time:

                    #### Define variables that are reset at each new loop ####
                    is_trade = False  # bool to determine if a trade can be made at a certain time based on the closest trades
                    time_diff = 16
                    closing = 0
                    benefit = "NA"
                    invest = False
                    decision = "WAIT"
                    current_time = datetime.datetime.fromtimestamp(time)
                    ####

                    #### Extract close price as the price of the closest trade in terms of time ####
                    while True:
                        if increment:
                            lineContent = close_price_file.readline().split("\t")
                        if float(lineContent[2]) < time - 15 :
                            increment = True  # Previously j += 1
                        elif float(lineContent[2]) >= time - 15 and float(lineContent[2]) <= time + 15:
                            is_trade = True
                            if abs(time - float(lineContent[2])) < time_diff:
                                time_diff = abs(time - float(lineContent[2]))
                                closing = float(lineContent[0])
                            increment = True  # Previously j += 1
                        elif float(lineContent[2]) > time + 15:
                            increment = False
                            break

                    if not is_trade:  # If no trade was found between the bracket, it means nobody responded to the trade and the trade is canceled
                        time += time_step
                        loop_nb += 1
                        with open("{}_{}/{}_records.txt".format(algorithm, benchmark, currency), "a") as balance_file:
                            balance_file.write("No trades found around time: {}\n".format(current_time))
                        continue
                    ####

                    #### Define decimal ceiling value ####
                    sum_decimal = utilities.decimal_round(decimal)
                    ####

                    #### Define base volume available for investment ####
                    base_volume = sum_to_invest[i]*float(funds[base_currency][i])  # base volume available for investment
                    ####

                    #### Evaluate the fee applied to the current volume ####
                    final_fee = utilities.eval_fee(fees, base_volume)
                    ####

                    #### Estimation of the investment amount per crypto #### Algorithm is here ####
                    if not price_last_decision[currency]:
                        order_price = round(float(closing) + sum_decimal, decimal)  # Define order price for buying higher than closing, and round to avoid weird decimals
                        volume_to_invest = utilities.volume_round(base_volume/order_price, order_price)  # Define volume to be invested in function of the order price and round it to avoid an excess of decimals due to python's poor handling of te floats
                        if len(history[currency]) > min_hist:  # If no purchase or sell was made, look at history
                            last_closing = [float(i[3]) for i in history[currency][-min_hist:]]
                            if order_price < statistics.median(last_closing):
                                decision = "buy"
                                invest = True

                    else:
                        if price_last_decision[currency][0] == "sell":  # buying algo
                            order_price = round(float(closing) + sum_decimal, decimal)  # Define order price for buying higher than closing, and round to avoid weird decimals
                            volume_to_invest = utilities.volume_round(base_volume/order_price, order_price)  # Define volume to be invested in function of the order price and round it to avoid an excess of decimals due to python's poor handling of te floats
                            last_closing = [float(i[3]) for i in history[currency][-min_hist:]]
                            if statistics.median(last_closing)*price_last_decision[currency][2] > min_benef*order_price*price_last_decision[currency][2]*(1+final_fee):  # If previous sell was 10% higher than current buy when accounting for fees, buy # note that previous volume needs to be used for a fair comparison !! PAY ATTENTION NOT TO USE IN THE ORDER
                                decision = "buy"
                                invest = True
                        elif price_last_decision[currency][0] == "buy":  # selling algo
                            order_price = round(float(closing) - sum_decimal, decimal)  # Define order price for selling lower than closing, and round to avoid weird decimals
                            volume_to_invest = price_last_decision[currency][2]
                            if len(history[currency]) > min_sell_history and len(history[currency]) > history_regression_sell:  # COMPLETE CONDITION BELOW WITH REGRESSION SLOPE AND MAKE CONDITION FAIL IF REGRESSION HISTORY INFERIOR TO THRESHOLD
                                last_closing = [float(i[3]) for i in history[currency][-min_sell_history:]]
                                last_time = [float(i[12]) for i in history[currency][-history_regression_sell:]]
                                coefficient = compute_linear_regression(last_time, [float(i[3]) for i in history[currency][-history_regression_sell:]])
                                if price_last_decision[currency][1]*price_last_decision[currency][2]*(1+price_last_decision[currency][3])*min_benef < order_price*price_last_decision[currency][2]*(1-final_fee):  # If previous buy was 10% lower than current sell when accounting for fees, sell
                                    if statistics.median(last_closing)*price_last_decision[currency][2] < order_price*price_last_decision[currency][2]*(1-final_fee):  # Second condition is about the sell median
                                        if coefficient < slope:  # If decrease is more important than threshold, sell
                                            decision = "sell"
                                            invest = True
                    ####

                    #### Evaluate if minimum volume is reached ####
                    if not utilities.is_min_vol(volume_to_invest, min_vol_order, currency):
                        continue
                    ####

                    #### Simulate purchase ####
                    if invest:
                        # Update balance and estimate benefit in case of sell
                        if decision == "buy":
                            funds[base_currency][i] -= order_price*volume_to_invest*(1+final_fee)
                            funds[currency] += volume_to_invest  # Here the funds of the currency are actually its volume
                        if decision == "sell":  # estimate benefit
                            funds[base_currency][i] += order_price*price_last_decision[currency][2]*(1-final_fee)
                            funds[currency] -= price_last_decision[currency][2]
                            benefit = order_price*price_last_decision[currency][2]*(1-final_fee) - price_last_decision[currency][1]*price_last_decision[currency][2]*(1+price_last_decision[currency][3])
                        # Write history file only in case of buy or sell
                        price_last_decision[currency] = [decision, order_price, volume_to_invest, final_fee]
                        with open("{}_{}/{}_last_activity.txt".format(algorithm, benchmark, currency), "w") as last_activity:
                            last_activity.write("{}\t{}\n".format("\t".join([str(i) for i in price_last_decision[currency]]), current_time))
                    ####

                    #### Write results in each respective file ####
                    history[currency].append([loop_nb, current_time, decision, order_price, decimal, final_fee, min_vol_order, volume_to_invest, funds[base_currency][i], funds[currency], closing, benefit, time])
                    with open("{}_{}/{}_records.txt".format(algorithm, benchmark, currency), "a") as balance_file:
                        balance_file.write("{}\n".format("\t".join([str(i) for i in history[currency][-1]])))
                    ####

                    #### Increment scaling variables ####
                    time += time_step
                    loop_nb += 1
                    ####

        # print("Finished benchmark on {}".format(currency))
        #### Redirect any exception into output files to have follow-up cycle by cycle ####
        except Exception as excpt:
            with open("{}_{}/{}_records.txt".format(algorithm, benchmark, currency), "a") as balance_file:
                balance_file.write("ERROR: {}\n".format(excpt))
            print("Error benchmark number {} : {}".format(param[2], excpt))
        ####
    print("Finished benchmark number {}".format(param[2]))

#### Start benchmark loop ####
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    pool_errors = executor.map(run_benchmark, all_param_comb)
print(pool_errors)
#### End investment loop ####

#### Print benchmark duration ####
print("Benchmark duration: {} minutes".format((time.time() - chrono_start)/60))
####
