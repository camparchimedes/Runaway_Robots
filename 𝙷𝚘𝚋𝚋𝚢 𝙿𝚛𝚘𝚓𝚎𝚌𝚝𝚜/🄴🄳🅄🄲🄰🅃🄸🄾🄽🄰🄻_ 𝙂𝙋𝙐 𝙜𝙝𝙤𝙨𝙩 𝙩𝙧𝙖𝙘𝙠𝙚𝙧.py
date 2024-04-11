﻿# @markdown 🄴🄳🅄🄲🄰🅃🄸🄾🄽🄰🄻: 𝙂𝙋𝙐 ghost 𝙩𝙧𝙖𝙘𝙠𝙚𝙧
import time
import math
from IPython.display import display, Javascript
import ipywidgets as widgets


def start_timer():
    global start_time
    start_time = time.time()


def end_timer():
    global end_time
    end_time = time.time()


def calculate_cost(runtime_minutes, cost_per_hour):
    cost_per_minute = cost_per_hour / 60
    total_cost = runtime_minutes * cost_per_minute
    return round(total_cost, 2)


def send_notification(cost):
    message = f"The current cost is ${cost} per hour."
    display(Javascript('alert("{msg}")'.format(msg=message)))


def main():
    cost_per_hour = float(input("Enter the cost per hour: "))
    while True:
        start_timer()
        time.sleep(60)  # sleep for one minute
        end_timer()
        runtime_minutes = math.ceil((end_time - start_time) / 60)
        total_cost = calculate_cost(runtime_minutes, cost_per_hour)
        if runtime_minutes % 60 == 0:  # send notification every hour
            send_notification(total_cost)


if __name__ == '__main__':
    main()
