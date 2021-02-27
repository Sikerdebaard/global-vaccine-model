from functools import lru_cache
from covid19.utils.vaccine import metadata_for_vaccine
from covid19.utils.country import country_vaccine_regimen

import pandas as pd


@lru_cache(maxsize=None)
def dose_regimen_for_vaccine(vaccine):
    df_metadata = metadata_for_vaccine(vaccine)

    assert 'doses' in df_metadata.index
    doses = int(df_metadata.loc['doses']['value'])
    if doses > 1:
        assert 'interval_between_doses' in df_metadata.index

        interval = int(df_metadata.loc['interval_between_doses']['value'])

        return {'doses': doses, 'interval': interval}

    return {'doses': doses}

@lru_cache(maxsize=None)
def is_vaccine_single_dose_regimen_for_country(vaccine):
    regimen = dose_regimen_for_vaccine(vaccine)

    # TODO: We should probably allow some country-specific parameters to override dose-regimen on specific vaccines
    # TODO: We should probably allow some country-specific parameters to override dose-regimen on specific vaccines
    # This should probably be implemented in dose_regimen_for_vaccine and/or vaccine_dose_intervals_for_country as well though

    return regimen['doses'] == 1

@lru_cache(maxsize=None)
def vaccine_dose_intervals_for_country(alpha3, vaccines_in_use, start_date, end_date):
    df_regimens = None
    for compounded_vaccine in vaccines_in_use:
        if df_regimens is not None and compounded_vaccine in df_regimens.columns:
            continue  # skip double vaccine mentions, no point in adding it again

        df_compounded = pd.DataFrame(columns=['interval'])

        for vaccine in compounded_vaccine:
            default_regimen = dose_regimen_for_vaccine(vaccine)

            if default_regimen['doses'] <= 1:
                continue

            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            df_regimen = pd.DataFrame(columns=['interval'], index=pd.date_range(start_date, end_date))

            df_country_regimen = country_vaccine_regimen(alpha3=alpha3)

            if df_country_regimen is not None:
                df_country_regimen = df_country_regimen[df_country_regimen['vaccine'] == vaccine]
                for idx, row in df_country_regimen.iterrows():
                    df_regimen.at[idx, 'interval'] = row['interval']

            # for idx, row in df_regimen.iterrows():
            #     if not pd.isna(df_regimen.at[idx, 'interval']):
            #         vals = df_regimen.at[idx, 'interval']
            #         if '-' in str(vals):
            #             vals = str(vals).split('-')
            #             df_regimen.at[idx, 'interval'] = f'{int(vals[0])-1}-{int(vals[1])+1}'

            if pd.isna(df_regimen.at[start_date, 'interval']):
                default = default_regimen['interval']
                if '-' not in str(default):
                    default = f'{int(default)-3}-{int(default)+3}'

                df_regimen.at[start_date, 'interval'] = default

            df_regimen = df_regimen.ffill()

            for idx, row in df_regimen.iterrows():
                if idx not in df_compounded.index:
                    df_compounded.at[idx, 'interval'] = row['interval']
                else:
                    intervals = [int(x) for x in row['interval'].split('-')] + [int(x) for x in df_compounded.at[idx, 'interval'].split('-')]
                    df_compounded.at[idx, 'interval'] = f'{min(intervals)}-{max(intervals)}'

        compounded_vaccine_name = '/'.join(compounded_vaccine)
        if df_regimens is None:
            df_regimens = df_compounded['interval'].rename(compounded_vaccine_name).to_frame()
        else:
            for idx, row in df_compounded.iterrows():
                #df_regimens = df_regimens.join(df_compounded['interval'].rename(compounded_vaccine_name))
                df_regimens.at[idx, compounded_vaccine_name] = df_compounded.at[idx, 'interval']

    return df_regimens.sort_index()

@lru_cache(maxsize=None)
def minmax_dose_intervals_for_country(alpha3, vaccines_in_use, start_date, end_date):
    df_regimen = vaccine_dose_intervals_for_country(alpha3=alpha3, vaccines_in_use=vaccines_in_use, start_date=start_date, end_date=end_date)

    df_minmax_regimen = pd.DataFrame(index=df_regimen.index)

    for col in df_regimen.columns:
        series_min = df_regimen[col].apply(lambda x: int(x.split('-')[0].strip()) if '-' in str(x) else x)
        series_max = df_regimen[col].apply(lambda x: int(x.split('-')[1].strip()) if '-' in str(x) else x)
        df_minmax_regimen[f'{col}_min'] = series_min
        df_minmax_regimen[f'{col}_max'] = series_max

    return df_minmax_regimen.astype(int)
