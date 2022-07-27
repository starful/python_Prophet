#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from prophet import Prophet


def main():
    # flights データセットを読み込む
    df = sns.load_dataset('flights')

    # カラムが年と月で分かれているのでマージする
    df['year-month'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str),
                                      format='%Y-%b')

    # Prophet が仮定するカラム名に変更する
    # タイムスタンプ: ds
    # 目的変数: y
    rename_mappings = {
        'year-month': 'ds',
        'passengers': 'y',
    }
    df.rename(columns=rename_mappings,
              inplace=True)

    # 不要なカラムを落とす
    df.drop(['year', 'month'],
            axis=1,
            inplace=True)

    # 時系列の順序で学習・検証用データをホールアウトする
    train_df, eval_df = train_test_split(df,
                                         shuffle=False,
                                         random_state=42,
                                         test_size=0.3)

    # 学習用データを使って学習する
    m = Prophet()
    m.fit(train_df)

    # 検証用データを予測する
    forecast = m.predict(eval_df.drop(['y'],
                                      axis=1))

    # 真の値との誤差を MAE で求める
    mae = mean_absolute_error(forecast['yhat'],
                              eval_df['y'])
    print(f'MAE: {mae:.05f}')

    # 実際のデータと予測をプロットする
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['ds'], df['y'], color='y')
    m.plot(forecast, ax=ax)
    # トレンドと季節成分をプロットする
    m.plot_components(forecast)

    plt.show()


if __name__ == '__main__':
    main()