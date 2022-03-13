import warnings
import os
import pandas as pd
import pytest
from zipline.utils.calendar_utils import get_calendar
import sqlalchemy as sa

from zipline.assets import (
    AssetDBWriter,
    AssetFinder,
    Equity,
    Future,
)


@pytest.fixture(scope="function")
def sql_db(request):
    url = "sqlite:///:memory:"
    request.cls.engine = sa.create_engine(url)
    yield request.cls.engine
    request.cls.engine.dispose()
    request.cls.engine = None


@pytest.fixture(scope="class")
def sql_db_class(request):
    url = "sqlite:///:memory:"
    request.cls.engine = sa.create_engine(url)
    yield request.cls.engine
    request.cls.engine.dispose()
    request.cls.engine = None


@pytest.fixture(scope="function")
def empty_assets_db(sql_db, request):
    AssetDBWriter(sql_db).write(None)
    request.cls.metadata = sa.MetaData(sql_db)
    request.cls.metadata.reflect(bind=sql_db)


@pytest.fixture(scope="class")
def with_default_date_bounds(request):
    request.cls.START_DATE = pd.Timestamp("2006-01-03", tz="utc")
    request.cls.END_DATE = pd.Timestamp("2006-12-29", tz="utc")


@pytest.fixture(scope="class")
def with_trading_calendars(request):
    """
    fixture providing cls.trading_calendar,
    cls.all_trading_calendars, cls.trading_calendar_for_asset_type as a
    class-level fixture.

    - `cls.trading_calendar` is populated with a default of the nyse trading
    calendar for compatibility with existing tests
    - `cls.all_trading_calendars` is populated with the trading calendars
    keyed by name,
    - `cls.trading_calendar_for_asset_type` is populated with the trading
    calendars keyed by the asset type which uses the respective calendar.

    Attributes
    ----------
    TRADING_CALENDAR_STRS : iterable
        iterable of identifiers of the calendars to use.
    TRADING_CALENDAR_FOR_ASSET_TYPE : dict
        A dictionary which maps asset type names to the calendar associated
        with that asset type.
    """

    request.cls.TRADING_CALENDAR_STRS = ("NYSE",)
    request.cls.TRADING_CALENDAR_FOR_ASSET_TYPE = {Equity: "NYSE", Future: "us_futures"}
    # For backwards compatibility, exisitng tests and fixtures refer to
    # `trading_calendar` with the assumption that the value is the NYSE
    # calendar.
    request.cls.TRADING_CALENDAR_PRIMARY_CAL = "NYSE"

    request.cls.trading_calendars = {}
    # Silence `pandas.errors.PerformanceWarning: Non-vectorized DateOffset
    # being applied to Series or DatetimeIndex` in trading calendar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for cal_str in set(request.cls.TRADING_CALENDAR_STRS) | {
            request.cls.TRADING_CALENDAR_PRIMARY_CAL
        }:
            # Set name to allow aliasing.
            calendar = get_calendar(cal_str)
            setattr(request.cls, "{0}_calendar".format(cal_str.lower()), calendar)
            request.cls.trading_calendars[cal_str] = calendar

        type_to_cal = request.cls.TRADING_CALENDAR_FOR_ASSET_TYPE.items()
        for asset_type, cal_str in type_to_cal:
            calendar = get_calendar(cal_str)
            request.cls.trading_calendars[asset_type] = calendar

    request.cls.trading_calendar = request.cls.trading_calendars[
        request.cls.TRADING_CALENDAR_PRIMARY_CAL
    ]


@pytest.fixture()
def logbook_activation_strategy():
    class ContextEnteringStrategy:
        def __init__(self, handler):
            self.handler = handler

        def activate(self):
            self.handler.__enter__()

        def deactivate(self):
            self.handler.__exit__(None, None, None)

        def __enter__(self):
            self.activate()
            return self.handler

        def __exit__(self, *_):
            self.deactivate()

    return ContextEnteringStrategy


@pytest.fixture(scope="class")
def set_test_bundle_core(request, tmpdir_factory):
    request.cls.environ = str(tmpdir_factory.mktemp("tmp"))


@pytest.fixture(scope="class")
def with_asset_finder(sql_db_class):
    def asset_finder(**kwargs):
        AssetDBWriter(sql_db_class).write(**kwargs)
        return AssetFinder(sql_db_class)

    return asset_finder
