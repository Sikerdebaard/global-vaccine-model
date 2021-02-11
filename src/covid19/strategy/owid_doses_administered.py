from covid19.utils.owid import country_vaccines_in_use, owid_vaccine_to_vaccine_name, country_data
from covid19.utils.dose_regimen import vaccine_dose_intervals_for_country
from covid19.utils.country import country_startdate
from modeling.simpleestimator import estimate_vaccinated_from_doses
from modeling.plotting.plotmodel import model_to_chart

import pandas as pd
import numpy as np

from pathlib import Path


def run(alpha3, outdir):
    # vaccination data for country + preprocess the data
    df_country = country_data(alpha3)
    df_country = _preprocess(df_country, alpha3)

    # get the vaccines in use by a specific country as reported by OWID
    vaccines_in_use = country_vaccines_in_use(alpha3=alpha3)

    # convert OWID vaccine names to vaccine names
    vaccines_in_use = tuple(set([owid_vaccine_to_vaccine_name(vacc) for vacc in vaccines_in_use]))

    df_dose_intervals = vaccine_dose_intervals_for_country(alpha3, vaccines_in_use, df_country.index[0], df_country.index[-1])

    # OWID dataset does not tell us how many dose of each vaccine, so we assume it's all doses for all vaccines and
    # then combine the uncertainty into a single estimate
    df_min_doses, df_max_doses = _minmaxdoses(df_country, vaccines_in_use)

    df_models = pd.DataFrame(index=df_country.index)
    for vaccine in vaccines_in_use:
        # df_max_doses is cumulative, but we need daily doses administered for our estimator
        maxdoses = df_max_doses[f'max_{vaccine}'].diff()
        maxdoses[0] = 0

        interval = list(df_dose_intervals[vaccine].values)

        vaccinated, fully_vaccinated, single_dose, _ = estimate_vaccinated_from_doses(maxdoses, interval=interval, cumulative_output=False)

        df_models['vaccinated', vaccine, 'max'] = vaccinated
        df_models['fully_vaccinated', vaccine, 'max'] = fully_vaccinated
        df_models['single_dose', vaccine, 'max'] = single_dose

    df_models.columns = pd.MultiIndex.from_tuples(df_models.columns)
    df_aggregated = _aggregate_models_minmax(df_models, df_country, vaccines_in_use)

    df_vaccinated = _combine_models(df_aggregated['vaccinated'].cumsum())
    df_fully_vaccinated = _combine_models(df_aggregated['fully_vaccinated'].cumsum())
    #df_single_dose = _combine_models(df_vaccinated - df_fully_vaccinated)
    df_single_dose = _combine_models(df_aggregated['single_dose'].cumsum())

    df_model = df_vaccinated.rename(columns={
        'mean': 'vaccinated',
        'min': 'vaccinated_min',
        'max': 'vaccinated_max',
    })

    df_model = df_model.join(df_fully_vaccinated.rename(columns={
        'mean': 'fully_vaccinated',
        'min': 'fully_vaccinated_min',
        'max': 'fully_vaccinated_max',
    }))

    df_model = df_model.join(df_single_dose.rename(columns={
        'mean': 'single_dose_vaccinated',
        'min': 'single_dose_vaccinated_min',
        'max': 'single_dose_vaccinated_max',
    }))

    df_model.index.rename('date', inplace=True)

    outdir = Path(outdir)
    model_csv = outdir / f'{alpha3}.csv'
    df_model.to_csv(model_csv)

    chart_file_out_path = outdir / f'{alpha3}.png'
    model_to_chart(df_model, df_country, chart_file_out_path, f'{alpha3}')

    return df_model

def _aggregate_models_minmax(df_model_outputs, df_country, vaccines_in_use):
    df_aggregated = pd.DataFrame(index=df_country.index)

    df_vaccine_startdates = _vaccine_startdate_mask_by_owid_startdate(df_country, vaccines_in_use)

    for dofor in df_model_outputs.columns.get_level_values(0):
        df_minmax = pd.DataFrame(index=df_country.index)
        df_model = df_model_outputs[dofor]

        for idx, row in df_model.iterrows():
            columns = df_vaccine_startdates.columns[df_vaccine_startdates.loc[idx]]
            selection = df_model.loc[idx, columns]
            df_minmax.at[idx, 'min'] = selection.min()
            df_minmax.at[idx, 'max'] = selection.max()

        df_aggregated[dofor, 'min'] = df_minmax['min'].values
        df_aggregated[dofor, 'max'] = df_minmax['max'].values

    df_aggregated.columns = pd.MultiIndex.from_tuples(df_aggregated.columns)
    return df_aggregated


def _combine_models(df_numbers):
    df_out = pd.DataFrame()
    for idx, row in df_numbers.iterrows():
        stats = row.describe()
        for col in stats.index:
            df_out.at[idx, col] = stats[col]

    return df_out[['min', 'max', 'mean']]


def _vaccine_startdate_mask_by_owid_startdate(df_country, vaccines_in_use):
    df_selected = pd.DataFrame(index=df_country.index)
    for vaccine in vaccines_in_use:
        for idx, row in df_country.iterrows():
            owid_vaccines = [owid_vaccine_to_vaccine_name(x.strip()) for x in row['vaccine'].split(',')]
            if vaccine in owid_vaccines:
                df_selected.at[idx, vaccine] = True
            else:
                df_selected.at[idx, vaccine] = False

    return df_selected


def _minmaxdoses(df_country, vaccines_in_use):
    df_min = pd.DataFrame(index=df_country.index)
    df_max = pd.DataFrame(index=df_country.index)

    df_selected = _vaccine_startdate_mask_by_owid_startdate(df_country, vaccines_in_use)

    for vaccine in vaccines_in_use:
        df_diff = df_country['total_vaccinations'].diff()

        colname = f'max_{vaccine}'
        selected_idx = df_selected[df_selected[vaccine] == True].index
        df_cumsum = df_diff.loc[selected_idx].cumsum().rename(colname).to_frame()
        df_cumsum.at[df_cumsum.index[0] - pd.Timedelta(days=1), colname] = 0
        df_cumsum.sort_index(inplace=True)
        df_max = df_max.join(df_cumsum)

    # assume linear growth for missing values
    df_max = df_max.interpolate('linear').ffill().fillna(0)

    columns = df_max.columns
    for col in columns:
        maxval = df_max.loc[:, df_max.columns != col].max(axis=1)
        colname = f'min_{col[4:]}'
        df_min[colname] = (df_max[col] - maxval).apply(lambda x: x if x >= 0 else 0)

    return df_min.astype(int), df_max.astype(int)


def _preprocess(df_country, alpha3):
    startdate = country_startdate(alpha3=alpha3)

    if startdate is None:
        startdate = df_country.index[0]
    else:
        startdate = pd.to_datetime(startdate)

    assert startdate <= df_country.index[0]
    zerodate = startdate - pd.Timedelta(days=1)

    if zerodate not in df_country.index:
        df_country.loc[zerodate] = None

    df_country.sort_index(inplace=True)

    intcols = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
    for col in intcols:
        if not df_country[col][1:].isna().any():
            df_country.at[zerodate, col] = 0

    # resample to daily interval + linear interpolate + ffill the gaps
    df_country = df_country.asfreq('D').interpolate('linear').ffill()

    for col in intcols:
        if not df_country[col].isna().any():
            df_country[col] = df_country[col].astype(int)

    #df_country.at[df_country.index[0], 'vaccine'] = df_country.at[df_country.index[1], 'vaccine']
    df_country['vaccine'] = df_country['vaccine'].ffill().bfill()

    return df_country
