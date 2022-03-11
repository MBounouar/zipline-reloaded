import warnings
import os
import numpy as np
import pandas as pd
import pytest
from zipline.utils.calendar_utils import get_calendar
from types import GetSetDescriptorType
from functools import partial
import sqlalchemy as sa

from zipline.assets import (
    Asset,
    AssetDBWriter,
    AssetFinder,
    Equity,
    ExchangeInfo,
    Future,
)
from zipline.assets.continuous_futures import CHAIN_PREDICATES
from zipline.utils.date_utils import make_utc_aware


@pytest.fixture(scope="function")
def set_test_write(request, tmp_path):
    request.cls.assets_db_path = path = os.path.join(
        str(tmp_path),
        "assets.db",
    )
    request.cls.writer = AssetDBWriter(path)


@pytest.fixture(scope="function")
def set_test_asset(request):
    # Dynamically list the Asset properties we want to test.
    request.cls.asset_attrs = [
        name
        for name, value in vars(Asset).items()
        if isinstance(value, GetSetDescriptorType)
    ]

    # Very wow
    request.cls.asset = Asset(
        1337,
        symbol="DOGE",
        asset_name="DOGECOIN",
        start_date=pd.Timestamp("2013-12-08 9:31", tz="UTC"),
        end_date=pd.Timestamp("2014-06-25 11:21", tz="UTC"),
        first_traded=pd.Timestamp("2013-12-08 9:31", tz="UTC"),
        auto_close_date=pd.Timestamp("2014-06-26 11:21", tz="UTC"),
        exchange_info=ExchangeInfo("THE MOON", "MOON", "??"),
    )

    request.cls.test_exchange = ExchangeInfo("test full", "test", "??")
    request.cls.asset3 = Asset(3, exchange_info=request.cls.test_exchange)
    request.cls.asset4 = Asset(4, exchange_info=request.cls.test_exchange)
    request.cls.asset5 = Asset(
        5,
        exchange_info=ExchangeInfo(
            "still testing",
            "still testing",
            "??",
        ),
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


@pytest.fixture(scope="function")
def set_test_asset_finder(sql_db, request):
    AssetDBWriter(sql_db).write(None)
    request.cls._asset_writer = AssetDBWriter(sql_db)
    request.cls.asset_finder = AssetFinder(sql_db)


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


@pytest.fixture(scope="class")
def equity_info():
    """make_equity_info"""
    T = partial(pd.Timestamp, tz="UTC")

    def asset(sid, symbol, start_date, end_date):
        return dict(
            sid=sid,
            symbol=symbol,
            start_date=T(start_date),
            end_date=T(end_date),
            exchange="NYSE",
        )

    records = [
        asset(1, "A", "2014-01-02", "2014-01-31"),
        asset(2, "A", "2014-02-03", "2015-01-02"),
        asset(3, "B", "2014-01-02", "2014-01-15"),
        asset(4, "B", "2014-01-17", "2015-01-02"),
        asset(5, "C", "2001-01-02", "2015-01-02"),
        asset(6, "D", "2001-01-02", "2015-01-02"),
        asset(7, "FUZZY", "2001-01-02", "2015-01-02"),
    ]
    return pd.DataFrame.from_records(records)


@pytest.fixture(scope="class")
def test_finance_equity_info():
    """make_equity_info test finance"""
    T = partial(pd.Timestamp, tz="UTC")

    def asset(sid, symbol, start_date, end_date):
        return dict(
            sid=sid,
            symbol=symbol,
            start_date=T(start_date),
            end_date=T(end_date),
            exchange="NYSE",
        )

    records = [
        asset(1, "A", "2006-01-03", "2006-12-29"),
        asset(2, "B", "2006-01-03", "2006-12-29"),
        asset(133, "C", "2006-01-03", "2006-12-29"),
    ]
    return pd.DataFrame.from_records(records)


@pytest.fixture(scope="class")
def set_test_vectorized_symbol_lookup(request, sql_db_class, equity_info):
    ASSET_FINDER_COUNTRY_CODE = "??"

    equities = equity_info
    exchange_names = [df["exchange"] for df in (equities,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    AssetDBWriter(sql_db_class).write(
        equities=equities,
        futures=None,
        exchanges=exchanges,
        root_symbols=None,
        equity_supplementary_mappings=None,
    )
    request.cls.asset_finder = AssetFinder(sql_db_class)


@pytest.fixture(scope="class")
def futures_info():
    """make_future_info"""
    return pd.DataFrame.from_dict(
        {
            2468: {
                "symbol": "OMH15",
                "root_symbol": "OM",
                "notice_date": pd.Timestamp("2014-01-20", tz="UTC"),
                "expiration_date": pd.Timestamp("2014-02-20", tz="UTC"),
                "auto_close_date": pd.Timestamp("2014-01-18", tz="UTC"),
                "tick_size": 0.01,
                "multiplier": 500.0,
                "exchange": "TEST",
            },
            0: {
                "symbol": "CLG06",
                "root_symbol": "CL",
                "start_date": pd.Timestamp("2005-12-01", tz="UTC"),
                "notice_date": pd.Timestamp("2005-12-20", tz="UTC"),
                "expiration_date": pd.Timestamp("2006-01-20", tz="UTC"),
                "multiplier": 1.0,
                "exchange": "TEST",
            },
        },
        orient="index",
    )


@pytest.fixture(scope="class")
def set_test_futures(request, sql_db_class, futures_info):
    ASSET_FINDER_COUNTRY_CODE = "??"

    futures = futures_info
    exchange_names = [df["exchange"] for df in (futures,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    AssetDBWriter(sql_db_class).write(
        equities=None,
        futures=futures,
        exchanges=exchanges,
        root_symbols=None,
        equity_supplementary_mappings=None,
    )
    request.cls.asset_finder = AssetFinder(sql_db_class)


@pytest.fixture(scope="class")
def set_test_finance(request, sql_db_class, test_finance_equity_info):
    ASSET_FINDER_COUNTRY_CODE = "??"

    equities = test_finance_equity_info
    exchange_names = [df["exchange"] for df in (equities,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    AssetDBWriter(sql_db_class).write(
        equities=equities,
        futures=None,
        exchanges=exchanges,
        root_symbols=None,
        equity_supplementary_mappings=None,
    )

    request.cls.asset_finder = AssetFinder(sql_db_class)


@pytest.fixture(scope="class")
def set_test_benchmark_spec(request, sql_db_class):
    ASSET_FINDER_COUNTRY_CODE = "??"
    START_DATE = pd.Timestamp("2006-01-03", tz="utc")
    END_DATE = pd.Timestamp("2006-12-29", tz="utc")
    request.cls.START_DATE = START_DATE
    request.cls.END_DATE = END_DATE

    zero_returns_index = pd.date_range(
        request.cls.START_DATE,
        request.cls.END_DATE,
        freq="D",
        tz="utc",
    )
    request.cls.zero_returns = pd.Series(index=zero_returns_index, data=0.0)

    equities = pd.DataFrame.from_dict(
        {
            1: {
                "symbol": "A",
                "start_date": START_DATE,
                "end_date": END_DATE + pd.Timedelta(days=1),
                "exchange": "TEST",
            },
            2: {
                "symbol": "B",
                "start_date": START_DATE,
                "end_date": END_DATE + pd.Timedelta(days=1),
                "exchange": "TEST",
            },
        },
        orient="index",
    )

    equities = equities
    exchange_names = [df["exchange"] for df in (equities,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    AssetDBWriter(sql_db_class).write(
        equities=equities,
        futures=None,
        exchanges=exchanges,
        root_symbols=None,
        equity_supplementary_mappings=None,
    )

    request.cls.asset_finder = AssetFinder(sql_db_class)


@pytest.fixture(scope="class")
def set_test_ordered_futures_contracts(request, sql_db_class):
    ASSET_FINDER_COUNTRY_CODE = "??"

    roots_symbols = pd.DataFrame(
        {
            "root_symbol": ["FO", "BA", "BZ"],
            "root_symbol_id": [1, 2, 3],
            "exchange": ["CMES", "CMES", "CMES"],
        }
    )

    fo_frame = pd.DataFrame(
        {
            "root_symbol": ["FO"] * 4,
            "asset_name": ["Foo"] * 4,
            "symbol": ["FOF16", "FOG16", "FOH16", "FOJ16"],
            "sid": range(1, 5),
            "start_date": pd.date_range("2015-01-01", periods=4, tz="UTC"),
            "end_date": pd.date_range("2016-01-01", periods=4, tz="UTC"),
            "notice_date": pd.date_range("2016-01-01", periods=4, tz="UTC"),
            "expiration_date": pd.date_range("2016-01-01", periods=4, tz="UTC"),
            "auto_close_date": pd.date_range("2016-01-01", periods=4, tz="UTC"),
            "tick_size": [0.001] * 4,
            "multiplier": [1000.0] * 4,
            "exchange": ["CMES"] * 4,
        }
    )
    # BA is set up to test a quarterly roll, to test Eurodollar-like
    # behavior
    # The roll should go from BAH16 -> BAM16
    ba_frame = pd.DataFrame(
        {
            "root_symbol": ["BA"] * 3,
            "asset_name": ["Bar"] * 3,
            "symbol": ["BAF16", "BAG16", "BAH16"],
            "sid": range(5, 8),
            "start_date": pd.date_range("2015-01-01", periods=3, tz="UTC"),
            "end_date": pd.date_range("2016-01-01", periods=3, tz="UTC"),
            "notice_date": pd.date_range("2016-01-01", periods=3, tz="UTC"),
            "expiration_date": pd.date_range("2016-01-01", periods=3, tz="UTC"),
            "auto_close_date": pd.date_range("2016-01-01", periods=3, tz="UTC"),
            "tick_size": [0.001] * 3,
            "multiplier": [1000.0] * 3,
            "exchange": ["CMES"] * 3,
        }
    )
    # BZ is set up to test the case where the first contract in a chain has
    # an auto close date before its start date. It also tests the case
    # where a contract in the chain has a start date after the auto close
    # date of the previous contract, leaving a gap with no active contract.
    bz_frame = pd.DataFrame(
        {
            "root_symbol": ["BZ"] * 4,
            "asset_name": ["Baz"] * 4,
            "symbol": ["BZF15", "BZG15", "BZH15", "BZJ16"],
            "sid": range(8, 12),
            "start_date": [
                pd.Timestamp("2015-01-02", tz="UTC"),
                pd.Timestamp("2015-01-03", tz="UTC"),
                pd.Timestamp("2015-02-23", tz="UTC"),
                pd.Timestamp("2015-02-24", tz="UTC"),
            ],
            "end_date": pd.date_range(
                "2015-02-01",
                periods=4,
                freq="MS",
                tz="UTC",
            ),
            "notice_date": [
                pd.Timestamp("2014-12-31", tz="UTC"),
                pd.Timestamp("2015-02-18", tz="UTC"),
                pd.Timestamp("2015-03-18", tz="UTC"),
                pd.Timestamp("2015-04-17", tz="UTC"),
            ],
            "expiration_date": pd.date_range(
                "2015-02-01",
                periods=4,
                freq="MS",
                tz="UTC",
            ),
            "auto_close_date": [
                pd.Timestamp("2014-12-29", tz="UTC"),
                pd.Timestamp("2015-02-16", tz="UTC"),
                pd.Timestamp("2015-03-16", tz="UTC"),
                pd.Timestamp("2015-04-15", tz="UTC"),
            ],
            "tick_size": [0.001] * 4,
            "multiplier": [1000.0] * 4,
            "exchange": ["CMES"] * 4,
        }
    )

    futures = pd.concat([fo_frame, ba_frame, bz_frame])

    exchange_names = [df["exchange"] for df in (futures,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    AssetDBWriter(sql_db_class).write(
        equities=None,
        futures=futures,
        exchanges=exchanges,
        root_symbols=roots_symbols,
        equity_supplementary_mappings=None,
    )

    request.cls.asset_finder = AssetFinder(sql_db_class)


@pytest.fixture(scope="class")
def set_test_adjustments(request, tmp_path_factory):
    request.cls.db_path = str(tmp_path_factory.mktemp("tmp") / "adjustments.db")


class ActivationStrategy(object):
    def __init__(self, handler):
        super(ActivationStrategy, self).__init__()
        self.handler = handler

    def activate(self):
        raise NotImplementedError()  # pragma: no cover

    def deactivate(self):
        raise NotImplementedError()  # pragma: no cover

    def __enter__(self):
        self.activate()
        return self.handler

    def __exit__(self, *_):
        self.deactivate()


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


# @pytest.fixture(params=[ContextEnteringStrategy])
# def logbook_activation_strategy(request=[ContexEnteringStrategy]):
# return request.param
@pytest.fixture()
def logbook_activation_strategy():
    return ContextEnteringStrategy
