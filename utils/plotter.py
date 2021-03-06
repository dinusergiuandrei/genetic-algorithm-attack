import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_per_generation():
    for file in os.listdir('history'):
        location = os.path.join('history', file)
        history = np.load(location)
        fname = os.path.splitext(file)[0]
        df = pd.DataFrame(history).T.melt()
        df.columns = ['generation', 'value']

        # _ = sns.lineplot(x='generation', y='value', data=df).set_title(fname)
        s = df.groupby('generation').min()
        ax = s.plot()
        _ = ax.set_title(str(s.min()))
        plt.savefig(f'plots/{fname}.png', format='PNG')
        plt.clf()
        print(fname, history.shape)


def plot_time():
    df = pd.read_csv('../time_result.csv')
    df['times'] = df.times
    sns.lineplot(x='workers', y='times', data=df)
    plt.savefig(f'plots/time_by_workers.png', format='PNG')


if __name__ == '__main__':
    plot_per_generation()
