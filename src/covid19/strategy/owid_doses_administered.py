from covid19.utils.owid import country_vaccines_in_use, owid_vaccine_to_vaccine_name, country_data, country_vaccine_startdates as owid_country_vaccine_startdates
from covid19.utils.dose_regimen import minmax_dose_intervals_for_country, vaccine_dose_intervals_for_country, is_vaccine_single_dose_regimen_for_country
from covid19.utils.country import country_startdate, country_doses_administered_by_vaccine, country_vaccine_startdate, first_second_dose_date
from modeling.simpleestimator import estimate_vaccinated_from_doses
from modeling.plotting.plotmodel import model_to_chart

import pandas as pd
import numpy as np

from pathlib import Path


def strategy_doses_per_vaccine(alpha3, outdir):
    df_country, vaccines_in_use, df_dose_intervals = _countrydata(alpha3)

def strategy_estimated_doses_per_vaccine(alpha3, outdir, df_country=None, title=None, subtitle=None):
    df_owid_country, vaccines_in_use, df_dose_intervals = _countrydata(alpha3)

    if df_country is None:
        print('Using OWID country data')
        df_country = df_owid_country
    else:
        print('Using custom country data')

    df_doses_by_vaccine = _estimate_doses_per_vaccine(country_doses_administered_by_vaccine(alpha3=alpha3), df_country, alpha3)
    df_minmax_vaccine_intervals = minmax_dose_intervals_for_country(alpha3, vaccines_in_use, df_country.index[0], df_country.index[-1])
    vaccine_interval_csv = outdir / f'{alpha3}-dose-intervals.csv'
    df_minmax_vaccine_intervals.round(0).astype(pd.Int64Dtype()).to_csv(vaccine_interval_csv)
    second_dose_date = first_second_dose_date(alpha3=alpha3)

    df_models = pd.DataFrame(index=df_country.index)
    for vaccine in vaccines_in_use:
        if vaccine not in df_doses_by_vaccine.columns:
            # TODO: Throw warning here, skip this vaccine for now
            continue

        if is_vaccine_single_dose_regimen_for_country(vaccine):
            print(f'Single dose {vaccine}')
            vaccinated = df_doses_by_vaccine[vaccine]
            fully_vaccinated = df_doses_by_vaccine[vaccine]
            started_regimen = df_doses_by_vaccine[vaccine]

            df_models['vaccinated', 'min', vaccine] = vaccinated
            df_models['fully_vaccinated', 'min', vaccine] = fully_vaccinated
            df_models['started_regimen', 'min', vaccine] = started_regimen

            df_models['vaccinated', 'max', vaccine] = vaccinated
            df_models['fully_vaccinated', 'max', vaccine] = fully_vaccinated
            df_models['started_regimen', 'max', vaccine] = started_regimen
        else:
            # doses is cumulative, but we need daily doses administered for our estimator
            doses = df_doses_by_vaccine[vaccine].diff()
            doses[0] = 0

            intervals_min = list(df_minmax_vaccine_intervals[f'{vaccine}_min'].values)
            intervals_max = list(df_minmax_vaccine_intervals[f'{vaccine}_max'].values)

            vaccinated, fully_vaccinated, started_regimen, _ = estimate_vaccinated_from_doses(doses, interval=intervals_min, cumulative_output=True)
            df_models['vaccinated', 'min', vaccine] = vaccinated
            df_models['fully_vaccinated', 'max', vaccine] = fully_vaccinated
            df_models['started_regimen', 'min', vaccine] = started_regimen

            vaccinated, fully_vaccinated, started_regimen, _ = estimate_vaccinated_from_doses(doses, interval=intervals_max, cumulative_output=True)
            df_models['vaccinated', 'max', vaccine] = vaccinated
            df_models['fully_vaccinated', 'min', vaccine] = fully_vaccinated
            df_models['started_regimen', 'max', vaccine] = started_regimen

    df_models.columns = pd.MultiIndex.from_tuples(df_models.columns)
    vaccine_model_csv = outdir / f'{alpha3}-vaccines.csv'
    df_models.round(0).astype(pd.Int64Dtype()).to_csv(vaccine_model_csv)

    df_aggregated = _sum_models_minmax(df_models)
    df_aggregated.astype(pd.Int64Dtype())

    if second_dose_date is not None:
        idx = df_aggregated[df_aggregated['fully_vaccinated', 'max'].index < second_dose_date].index
        df_aggregated.loc[idx, 'fully_vaccinated'] = 0

    df_vaccinated = _combine_models(df_aggregated['vaccinated'])
    df_fully_vaccinated = _combine_models(df_aggregated['fully_vaccinated'])
    df_started_regimen = _combine_models(df_aggregated['started_regimen'])

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

    df_model = df_model.join(df_started_regimen.rename(columns={
        'mean': 'single_dose_vaccinated',
        'min': 'single_dose_vaccinated_min',
        'max': 'single_dose_vaccinated_max',
    }))

    df_model.index.rename('date', inplace=True)

    outdir = Path(outdir)
    model_csv = outdir / f'{alpha3}.csv'
    df_model.round(0).astype(pd.Int64Dtype()).to_csv(model_csv)

    chart_file_out_path = outdir / f'{alpha3}.png'

    if title is None:
        title = f'{alpha3}'

    model_to_chart(df_model, df_country, chart_file_out_path, title=title, subtitle=subtitle)

    return df_model

def _sum_models_minmax(df_model_outputs):
    df_aggregated = pd.DataFrame(index=df_model_outputs.index)

    for dofor in df_model_outputs.columns.get_level_values(0):
        df_aggregated[dofor, 'min'] = df_model_outputs[dofor, 'min'].sum(axis=1)
        df_aggregated[dofor, 'max'] = df_model_outputs[dofor, 'max'].sum(axis=1)

    df_aggregated.columns = pd.MultiIndex.from_tuples(df_aggregated.columns)
    return df_aggregated

def _startdateforvaccine(vaccine, alpha3):
    date = country_vaccine_startdate(vaccine, alpha3=alpha3)

    if date is not None:
        return pd.to_datetime(date)

    vaccine_startdates = owid_country_vaccine_startdates(alpha3=alpha3)

    for owid_vaccine, startdate  in vaccine_startdates.items():
        vaccine_name = owid_vaccine_to_vaccine_name(owid_vaccine)
        if vaccine_name == vaccine:
            return startdate

def _estimate_doses_per_vaccine(df_doses_by_vaccine, df_country, alpha3):
    df_doses_by_vaccine = df_doses_by_vaccine.pivot_table(index='date', values=['cumulative_administered'], columns='vaccine')
    df_doses_by_vaccine.columns = [x[1] for x in df_doses_by_vaccine.columns]

    vaccination_startdate = pd.to_datetime(country_startdate(alpha3=alpha3))  - pd.Timedelta(days=1)

    if vaccination_startdate not in df_doses_by_vaccine.index:
        df_doses_by_vaccine.loc[vaccination_startdate] = 0
        df_doses_by_vaccine.sort_index(inplace=True)

    df_doses_by_vaccine = df_doses_by_vaccine.asfreq('D')

    for col in df_doses_by_vaccine.columns:
        startdate = pd.to_datetime(_startdateforvaccine(col, alpha3)) - pd.Timedelta(days=1)
        df_doses_by_vaccine.at[startdate, col] = 0

    df_doses_by_vaccine.sort_index(inplace=True)

    df_doses_by_vaccine = df_doses_by_vaccine.interpolate('linear').ffill().round(0).astype(pd.Int64Dtype())

    df_doses_by_vaccine['sum'] = df_doses_by_vaccine.sum(axis=1).astype(int)

    df_doses_by_vaccine_pct = df_doses_by_vaccine.copy()
    for col in df_doses_by_vaccine_pct.columns:
        if col == 'sum':
            continue

        df_doses_by_vaccine_pct[col] = df_doses_by_vaccine_pct[col] / df_doses_by_vaccine_pct['sum']

    df_doses_by_vaccine_pct.drop(columns=['sum'], inplace=True)
    df_doses_by_vaccine_pct['sum'] = df_doses_by_vaccine_pct.sum(axis=1)

    df_doses_by_vaccine_pct.loc[df_doses_by_vaccine_pct.index[0]] = 0
    df_doses_by_vaccine_pct.at[df_doses_by_vaccine_pct.index[0], 'sum'] = 1.0

    df_estimate = pd.DataFrame(index=df_country.index)
    for col in df_doses_by_vaccine_pct.columns:
        if col == 'sum':
            continue

        df_estimate[col] = df_doses_by_vaccine_pct[col]
        df_estimate[col] = df_estimate[col].ffill()
        df_estimate[col] = (df_country['total_vaccinations'] * df_estimate[col]).astype(int)

    return df_estimate

def strategy_total_doses_only(alpha3, outdir):
    df_country, vaccines_in_use, df_dose_intervals = _countrydata(alpha3)

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

def _countrydata(alpha3):
    # vaccination data for country + preprocess the data
    df_country = country_data(alpha3)
    df_country = _preprocess(df_country, alpha3)

    # get the vaccines in use by a specific country as reported by OWID
    vaccines_in_use = country_vaccines_in_use(alpha3=alpha3)

    # convert OWID vaccine names to vaccine names
    vaccines_in_use = tuple(set([owid_vaccine_to_vaccine_name(vacc) for vacc in vaccines_in_use]))

    # dose intervals, customized to country data (if available, else assume defaults)
    df_dose_intervals = minmax_dose_intervals_for_country(alpha3, vaccines_in_use, df_country.index[0], df_country.index[-1])

    return df_country, vaccines_in_use, df_dose_intervals

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
    df_country = df_country.asfreq('D')
    df_country['total_vaccinations'] = df_country['total_vaccinations'].interpolate('linear').ffill()

    for col in intcols:
        if not df_country[col].isna().any():
            df_country[col] = df_country[col].astype(int)

    #df_country.at[df_country.index[0], 'vaccine'] = df_country.at[df_country.index[1], 'vaccine']
    df_country['vaccine'] = df_country['vaccine'].ffill().bfill()

    return df_country
