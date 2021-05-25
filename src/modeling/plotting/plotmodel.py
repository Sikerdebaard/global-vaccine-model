# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np


def _y_fmt_human(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9]
    suffix = ["G", "M", "k", "", "m", "u", "n"]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >= d:
            val = y / float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return f'{int(val)} {suffix[i]}'
            else:
                if signf == 1:
                    if str(val).split(".")[1] == "0":
                        return f'{int(val)} {suffix[i]}'
                tx = "{" + "val:.{signf}f".format(signf=signf) + "} {suffix}"
                return tx.format(val=val, suffix=suffix[i])

    return y


def model_to_chart(df_model, df_regionvals, chart_file_out_path, title, subtitle=None, plot_realworld=True, dpi=300):
    ignore = ('min', 'max')
    use_cols = [x for x in df_model.columns if not any([y in x for y in ignore]) and x != 'vaccinated']

    df_filter = df_model.copy()
    cols_filter = [x for x in df_model.columns if 'fully' in x]

    lastidx = None
    for idx, row in df_filter[cols_filter].iterrows():
        if row.sum() == 0:
            lastidx = idx
            df_filter.loc[idx, cols_filter] = None
    df_filter.loc[lastidx, cols_filter] = 0

    df_filter = df_filter.join(df_regionvals['total_vaccinations'])

    if plot_realworld:
        realworldcols = []
        legend = ["Total doses administered  (estimate)", "Prediction people fully vaccinated", "Prediction people started regimen"]
        if 'people_vaccinated' in df_regionvals.columns and df_regionvals['people_vaccinated'].sum() > 0:
            realworldcols.append('people_vaccinated')
            legend.append('people_vaccinated')
        if 'people_fully_vaccinated' in df_regionvals.columns  and df_regionvals['people_fully_vaccinated'].sum() > 0:
            realworldcols.append('people_fully_vaccinated')
            legend.append('RIVM Fully vaccinated (estimate)')

        df_filter = df_filter.join(df_regionvals[realworldcols])
        cols = ['total_vaccinations', *use_cols, *realworldcols]
    else:
        cols = ['total_vaccinations', *use_cols]
        legend = ["Total doses administered", "Fully vaccinated", "Started regimen"]

    df_filter.index = df_filter.index.date

    style = ['o'] + (['-'] * 2)
    colors = [
        [.6, .6, .6],
        [0, .7, .1],
        [1, .5, .1],
    ]
    if plot_realworld:
        if 'people_vaccinated' in realworldcols:
            colors.append([1, 0, 0])
            style.append('o')
        if 'people_fully_vaccinated' in realworldcols:
            colors.append([0.66, 0.55, 0.84])
            style.append('o')

    params = dict(
        figsize=(12, 9),
        alpha=.7,
        color=colors,
        grid=True,
        lw=3,
        fontsize=12,
        style=style,
    )

    ax = df_filter[cols].plot(**params)
    ax.legend(legend, loc='upper left')

    fill_colors = {
        'fully_vaccinated': (0, .8, 0),
        'single_dose_vaccinated': (1, .7, 0),
    }

    for col in use_cols:
        plt.fill_between(df_filter.index, df_filter[f'{col}_min'], df_filter[f'{col}_max'], color=fill_colors[col], alpha=.2)

    ax.set_ylabel(ylabel='Number of people vaccinated')
    ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt_human))
    fig = ax.get_figure()

    ax.set_xticks([])
    ax.set_xticks([], minor=True)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=15)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    if title:
        mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
        plt.suptitle(title, x=mid, y=.95, fontsize=16)

    if subtitle:
        ax.set_title(subtitle, fontsize=12)

    ax.margins(x=0, y=0)

    fig.savefig(chart_file_out_path, dpi=dpi)

    plt.close()
