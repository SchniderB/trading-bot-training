# -*- coding: utf-8 -*-
"""
Script that ranks all the benchmarks of a same algorithm number from the highest
to the lowest benefit.

Created on Sat Jul 18 15:28:53 2020

@author: boris
"""

import time
import datetime
import math
import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style

class Evaluation:
    """
    Contains different methods that evaluate the performances of the different
    benchmarks of a same algorithm.
    """
    def __init__(self):
        self.total_per_benchmark = []  # Total benefit of all cryptos together per benchmark
        self.top10_per_benchmark = []  # Benefit of the top 10 cryptos per benchmark
        self.starting_EUR_amount = 1000  # Manually define the starting amount
        self.algorithm_nb = "algorithm5"  # Manually define the algorithm of interest
        self.pipeline_parameters = dict()
        self.list_cryptos = []

    def benefit_per_crypto(self, crypto_ID):
        """
        Method that extracts the total benefit of the investment on a specific cryptocurrency
        for a specific benchmark.
        """
        # Extract last balance for a specific crypto
        last_line = []  # Only interested in the last line of the file that contains the last balance
        u = 0  # Counter for first time available
        start_time = 0
        end_time = 0
        with open("{}_records.txt".format(crypto_ID), "r") as input_file:
            input_file.readline()  # Skip header
            for line in input_file:
                if not "No trades" in line and not "ERROR" in line and not "Not enough crypto" in line:
                    last_line = line.split("\t")
                    if u == 0:
                        start_time = datetime.datetime.strptime(last_line[1], '%Y-%m-%d %H:%M:%S.%f').timestamp()  # Define start time crypto per crypto
                    u += 1
                    end_time = last_line[1]  # Every new line with any content is the new end_time so that the last one will be kept
        # Save total duration
        if end_time: end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f').timestamp()
        total_duration = (end_time - start_time) /  (3600 * 24)  # Convert total duration into days
        # Verify that the last balance does not include any crypto, and otherwise convert crypto in EUR
        if last_line: total = float(last_line[8])  # define total EUR per crypto
        if last_line and total_duration:  # If no trades, last_line will be empty and the following code will raise an error
            last_line[8] = float(last_line[8])  # Instead of total amount
            if float(last_line[9]) > 0.0:
                last_line[8] += float(last_line[9])*float(last_line[10])
                total += float(last_line[9])*float(last_line[10])
            last_line[8] = (last_line[8] - self.starting_EUR_amount) / total_duration
        else:
            return [0, crypto_ID, 0]

        if last_line: total -= self.starting_EUR_amount
        # Return [total_EUR, crypto_ID]
        return [last_line[8], crypto_ID, total]

    def benefit_per_benchmark(self, benchmark_ID):
        """
        Method that extracts the total benefit of the investment of a specific set
        of parameters, i.e. for a whole specific benchmark.
        """
        # Move to specific benchmark folder
        os.chdir("{}_{}".format(self.algorithm_nb, benchmark_ID))
        # Extract all crypto names from file names
        list_cryptos = [file.split("_")[0] for file in os.listdir(".") if "records" in file]
        # Write once list of crypto if is empty
        if not self.list_cryptos:
            self.list_cryptos = list_cryptos
        # Extract total money per crypto for the whole benchmark and generate top 10 cryptos
        print(benchmark_ID)
        total_all_cryptos = sorted([self.benefit_per_crypto(crypto) for crypto in list_cryptos], reverse=True)
        total_top10_cryptos = total_all_cryptos[:10]
        # Extract parameter file
        self.pipeline_parameters[benchmark_ID] = self.extract_param_per_benchmark()
        # Return to previous folder
        os.chdir("..")
        # Return the list of benefit per benchmark for all cryptos vs top 10
        self.total_per_benchmark.append([self.average_benefit(total_all_cryptos), benchmark_ID, total_all_cryptos])
        self.top10_per_benchmark.append([self.average_benefit(total_top10_cryptos), benchmark_ID, total_top10_cryptos])

    def average_benefit(self, list_of_lists):
        """
        Method that estimates the average benefit per day for a whole benchmark.
        """
        total = 0
        for sub_list in list_of_lists:
            total += sub_list[0]
        return total / len(list_of_lists)

    def extract_param_per_benchmark(self):
        """
        Method that extracts the list of parameters for a specific benchmark and
        stores it in a dictionary.
        """
        with open("pipeline_parameters.txt", "r") as param_file:
            param_file.readline()  # Skip header
            return param_file.readline().split("\t")

    def main(self):
        """
        Main method that runs all the other methods of the class Evaluation in order
        to perform the benchmark ranking.
        """
        # List all benchmarks of the algorithm of interest
        all_benchmarks = [directory.split("_")[1] for directory in os.listdir(".") if "{}_benchmark".format(self.algorithm_nb) in directory]
        # Extract total benefit per benchmark for all benchmarks
        for benchmark in all_benchmarks:
            self.benefit_per_benchmark(benchmark)
        # Sort the benchmark from best to worst
        self.total_per_benchmark = sorted(self.total_per_benchmark, reverse=True)
        self.top10_per_benchmark = sorted(self.top10_per_benchmark, reverse=True)
        # Perform weighted ranking with 50% of weight from all cryptos and 50% of weight from top 10 cryptos
        summed_ranks = {benchmark[1]:[i+1, benchmark[1], i+1, 0, benchmark[2]] for i, benchmark in enumerate(self.total_per_benchmark)}
        for i, benchmark in enumerate(self.top10_per_benchmark):
            summed_ranks[benchmark[1]][0] += i+1  # Rank tot + rank top10
            summed_ranks[benchmark[1]][3] += i+1  # 0 + rank top10
        # Write list with the final order
        final_ranks = sorted([summed_ranks[benchmark] for benchmark in summed_ranks.keys()])  # Sorting by the summed ranks increasingly
        # Write final output
        with open("{}_ranked_results.txt".format(self.algorithm_nb), "w") as output_file:
            output_file.write("weighted_rank\tbenchmark_number\ttotal_benefit_rank\ttop10_benefit_rank\tprice_check_frequency\tminimal_benefit\tminimal_history_size\tstarting_EUR_amount\t{}\n".format("\t".join(["crypto{}".format(i) for i in range(len(self.list_cryptos))])))
            for i, benchmark in enumerate(final_ranks):
                output_file.write("{}\n".format("\t".join(self.write_results(benchmark, i))))

        # Generate output directory for plots if does not already exist
        os.makedirs(self.algorithm_nb, exist_ok=True)

        # Generate plot of the benefit per day and total benefit
        self.barplot_benefit(final_ranks)
        self.barplot_total(final_ranks)

        # Generate plots of the balance evolution per crypto
        crypto_order = [str(crypto[1]) for crypto in final_ranks[0][4]]
        for crypto in crypto_order:
            self.line_plot_trades(final_ranks, crypto)

    def write_results(self, final_ranks, i):
        """
        Method to write the benchmark ranking in a joint file.
        """
        output = [str(val) for val in final_ranks[:4]]
        output[0] = str(i+1)  # Rank is currently the result of a sum of 2 ranks and is therefore not continuous, replace the value by real rank
        output.extend(self.pipeline_parameters[final_ranks[1]][:3])  # Extract pipeline parameters
        output.append(str(self.starting_EUR_amount))
        output.extend(["{}: {},{}".format(str(crypto[1]), str(crypto[0]), str(crypto[2])) for crypto in final_ranks[4]])  # str(crypto[4][2] is for the total gain and was added afterwards
        return output

    def barplot_benefit(self, final_ranks):
        """
        Method that generates a barplot for the top 10 results of the benchmark.
        """
        #Making the plot
        style.use('ggplot')

        # Re-organise data
        benchmark_list = [str(benchmark[1])[9:] for i,benchmark in enumerate(final_ranks) if i < 10]  # [9:] is to remove benchmark from the benchmark numbers
        crypto_order = [str(crypto[1]) for crypto in final_ranks[0][4]]
        gain_per_crypto = {crypto:[] for crypto in crypto_order}
        loss_per_crypto = {crypto:[] for crypto in crypto_order}
        net_benefit_per_crypto = [0 for benchmark in benchmark_list]
        is_total = False
        for crypto in crypto_order:
            for i, benchmark in enumerate(final_ranks):
                if i < 10:  # Only interested in the first 10 benchmarks which are the 10 most successful
                    for crpt in benchmark[4]:
                        if crypto in crpt:
                            if crpt[0] >= 0:
                                gain_per_crypto[crypto].append(crpt[0])
                                loss_per_crypto[crypto].append(0)
                            else:
                                gain_per_crypto[crypto].append(0)
                                loss_per_crypto[crypto].append(abs(crpt[0]))
                            if is_total:  # Only once net_benefit_per_crypto has been filled the break can be used
                                break
                        if not is_total:
                            net_benefit_per_crypto[i] += crpt[0]
                else:
                    break
            is_total = True  # Once cycle through all benchmarks will fill net_benefit_per_crypto

        fig, axs = plt.subplots(3, sharex=True)
        # fig.suptitle('Distribution of the trading prices and frequencies\nof {} across time'.format(pair))
        summed_gains = [0 for benchmark in benchmark_list]
        summed_losses = [0 for benchmark in benchmark_list]
        for i,crypto in enumerate(crypto_order):
            axs[0].bar(range(len(benchmark_list)), gain_per_crypto[crypto], label=crypto_order[i], bottom=summed_gains)
            axs[1].bar(range(len(benchmark_list)), loss_per_crypto[crypto], label=crypto_order[i], bottom=summed_losses)
            for j in range(len(summed_gains)):
                summed_gains[j] += gain_per_crypto[crypto][j]
                summed_losses[j] += loss_per_crypto[crypto][j]
        axs[2].bar(range(len(benchmark_list)), net_benefit_per_crypto, color="gray")

        lgd = axs[0].legend(bbox_to_anchor=(1.01, 1), ncol=2, prop={'size': 8})  # bbox is to position legend outside of the plot

        plt.xlabel('Benchmark number')
        plt.xticks(range(len(benchmark_list)), benchmark_list, fontsize=8)
        # # plt.ylabel('Trades per minute')
        axs[0].set_ylabel('Average gains\nper day [EUR]', fontsize = 10)
        axs[1].set_ylabel('Average losses\nper day [EUR]', fontsize = 10)
        axs[2].set_ylabel('Net benefit\nper day [EUR]', fontsize = 10)

        #plt.show()
        plt.savefig("{0}/{0}_benefit.png".format(self.algorithm_nb), dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Two bbox param are to resize final figure to avoid having the legend out of the figure
        plt.clf()  #clears the figure to be able to write the next plot
        plt.cla()
        plt.close()

    def barplot_total(self, final_ranks):
        """
        Method that generates a barplot for the top 10 results of the benchmark with
        the gain, losses and net benefit over all time period.
        """
        #Making the plot
        style.use('ggplot')

        # Re-organise data
        benchmark_list = [str(benchmark[1])[9:] for i,benchmark in enumerate(final_ranks) if i < 10]  # [9:] is to remove benchmark from the benchmark numbers
        crypto_order = [str(crypto[1]) for crypto in final_ranks[0][4]]
        gain_per_crypto = {crypto:[] for crypto in crypto_order}
        loss_per_crypto = {crypto:[] for crypto in crypto_order}
        net_benefit_per_crypto = [0 for benchmark in benchmark_list]
        is_total = False
        for crypto in crypto_order:
            for i, benchmark in enumerate(final_ranks):
                if i < 10:  # Only interested in the first 10 benchmarks which are the 10 most successful
                    for crpt in benchmark[4]:
                        if crypto in crpt:
                            if crpt[2] >= 0:
                                gain_per_crypto[crypto].append(crpt[2])
                                loss_per_crypto[crypto].append(0)
                            else:
                                gain_per_crypto[crypto].append(0)
                                loss_per_crypto[crypto].append(abs(crpt[2]))
                            if is_total:  # Only once net_benefit_per_crypto has been filled the break can be used
                                break
                        if not is_total:
                            net_benefit_per_crypto[i] += crpt[2]
                else:
                    break
            is_total = True  # Once cycle through all benchmarks will fill net_benefit_per_crypto

        fig, axs = plt.subplots(3, sharex=True)
        # fig.suptitle('Distribution of the trading prices and frequencies\nof {} across time'.format(pair))
        summed_gains = [0 for benchmark in benchmark_list]
        summed_losses = [0 for benchmark in benchmark_list]
        for i,crypto in enumerate(crypto_order):
            axs[0].bar(range(len(benchmark_list)), gain_per_crypto[crypto], label=crypto_order[i], bottom=summed_gains)
            axs[1].bar(range(len(benchmark_list)), loss_per_crypto[crypto], label=crypto_order[i], bottom=summed_losses)
            for j in range(len(summed_gains)):
                summed_gains[j] += gain_per_crypto[crypto][j]
                summed_losses[j] += loss_per_crypto[crypto][j]
        axs[2].bar(range(len(benchmark_list)), net_benefit_per_crypto, color="gray")  # Range of benchmark lenght is used to keep initial order

        lgd = axs[0].legend(bbox_to_anchor=(1.01, 1), ncol=2, prop={'size': 8})  # bbox is to position legend outside of the plot

        plt.xlabel('Benchmark number')
        plt.xticks(range(len(benchmark_list)), benchmark_list, fontsize=8)
        # # plt.ylabel('Trades per minute')
        axs[0].set_ylabel('Total gains [EUR]', fontsize = 10)
        axs[1].set_ylabel('Total losses [EUR]', fontsize = 10)
        axs[2].set_ylabel('Net benefit [EUR]', fontsize = 10)

        #plt.show()
        plt.savefig("{0}/{0}_total.png".format(self.algorithm_nb), dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Two bbox param are to resize final figure to avoid having the legend out of the figure
        plt.clf()  #clears the figure to be able to write the next plot
        plt.cla()
        plt.close()

    def line_plot_trades(self, final_ranks, crypto):
        """
        Method that plots the EUR balance after each SELL transaction.
        """
        # Extract list of top 5 benchmarks
        benchmark_list = [str(benchmark[1]) for i,benchmark in enumerate(final_ranks) if i < 5]
        # Extract all trades of the specific cryptocurrency for the top5 benchmarks
        sell_trade_history = {benchmark:[[], []] for benchmark in benchmark_list}
        for benchmark in benchmark_list:
            with open("{}_{}/{}_records.txt".format(self.algorithm_nb, benchmark, crypto), "r") as input_file:
                input_file.readline()
                for line in input_file:
                    if "sell" in line:
                        lineContent = line.split("\t")
                        sell_trade_history[benchmark][0].append(datetime.datetime.strptime(lineContent[1], '%Y-%m-%d %H:%M:%S.%f'))
                        sell_trade_history[benchmark][1].append(float(lineContent[8]))
                    elif "WAIT" in line:  # To keep a record of the last balance
                        lineContent = line.split("\t")
            # Save last amount even if no trades
            sell_trade_history[benchmark][0].append(datetime.datetime.strptime(lineContent[1], '%Y-%m-%d %H:%M:%S.%f'))
            lineContent[8] = float(lineContent[8])
            if float(lineContent[9]) > 0.0:
                lineContent[8] += float(lineContent[9])*float(lineContent[10])
            sell_trade_history[benchmark][1].append(lineContent[8])

        # Generate plot
        for benchmark in benchmark_list:
            plt.plot(sell_trade_history[benchmark][0], sell_trade_history[benchmark][1], label=benchmark[9:])

        lgd = plt.legend(bbox_to_anchor=(1.01, 1), ncol=1, prop={'size': 8})  # bbox is to position legend outside of the plot

        plt.xlabel('Time')
        plt.xticks(fontsize=8)
        plt.ylabel('Balance [EUR]')

        #plt.show()
        plt.savefig("{}/{}_balance_evolution.png".format(self.algorithm_nb, crypto), dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')  # Two bbox param are to resize final figure to avoid having the legend out of the figure
        plt.clf()  #clears the figure to be able to write the next plot
        plt.cla()
        plt.close()


if __name__ == '__main__':
    eval = Evaluation()
    resp = eval.main()
