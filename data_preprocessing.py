#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : data_preprocessing.py
@Author: XuYaoJian
@Date  : 2022/7/27 22:10
@Desc  : 
"""
import numpy as np
import pandas as pd

# 把原先为5分钟间隔的，改成间隔为10分钟的。隔一个再取值
def week_data():
    filename = 'data/week_data/California_autumn_20121001-20121007.csv'
    df = pd.read_csv(filename)
    wind_speed = []
    wind = df['wind speed at 100m (m/s)'].values
    datetime = pd.date_range(start="2012-10-01 00:00:00", end="2012-10-07 23:50:00", freq='10T')
    for index, value in enumerate(wind):
        if index % 2 == 0:
            wind_speed.append(wind[index])

    new_df = pd.DataFrame(data={
        'wind speed at 100m (m/s)': wind_speed,
        'datetime': datetime
    }, columns=['datetime', 'wind speed at 100m (m/s)'])
    new_df.to_csv('data/week_data/California_autumn_20121001-20121007新.csv', index=False)


if __name__ == "__main__":
    week_data()
