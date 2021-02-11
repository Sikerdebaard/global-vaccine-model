from covid19.strategy.owid_doses_administered import run as owid_run
from covid19.utils.owid import vaccinations_for_country


def model_country(country_iso_alpha3, outdir):
    strategy = _strategyfactory(country_iso_alpha3)
    df_model = strategy(country_iso_alpha3, outdir)

    return df_model


def _strategyfactory(alpha3):
    df_owid = vaccinations_for_country(alpha3=alpha3)

    # if df_owid['people_vaccinated'].dropna().sum() > 0:
    #     return _strategy_nopred

    return owid_run


def _strategy_nopred(alpha3):
    df_owid = vaccinations_for_country(alpha3=alpha3)
    return df_owid
