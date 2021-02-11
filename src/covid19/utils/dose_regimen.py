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
def vaccine_dose_intervals_for_country(alpha3, vaccines_in_use, start_date, end_date):
    df_regimens = None
    for vaccine in vaccines_in_use:
        if df_regimens is not None and vaccine in df_regimens.columns:
            continue  # skip double vaccine mentions, no point in adding it again

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

        if pd.isna(df_regimen.at[start_date, 'interval']):
            df_regimen.at[start_date, 'interval'] = default_regimen['interval']

        df_regimen = df_regimen.ffill()



        if df_regimens is None:
            df_regimens = df_regimen['interval'].rename(vaccine).to_frame()
            df_regimens.to_csv('/tmp/last.csv')
        else:
            df_regimens = df_regimens.join(df_regimen['interval'].rename(vaccine))
            df_regimens.to_csv('/tmp/last.csv')

    return df_regimens
