from covid19.strategy.owid_doses_administered import strategy_total_doses_only, strategy_estimated_doses_per_vaccine
from covid19.utils.owid import vaccinations_for_country


def model_country(country_iso_alpha3, outdir, df_country=None):
    strategy = _strategyfactory(country_iso_alpha3)
    df_model = strategy(country_iso_alpha3, outdir, df_country=df_country)

    return df_model


def _strategyfactory(alpha3):
    df_owid = vaccinations_for_country(alpha3=alpha3)

    # if df_owid['people_vaccinated'].dropna().sum() > 0:
    #     return _strategy_nopred

    #return owid_run

    return strategy_estimated_doses_per_vaccine


def _strategy_nopred(alpha3):
    df_owid = vaccinations_for_country(alpha3=alpha3)
    return df_owid
