import attr
from attr import validators
import pandas as pd
from pandas import NaT

from zipline.utils.calendar_utils import TradingCalendar

from zipline.data.bar_reader import OHLCV, NoDataOnDate, NoDataForSid
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.input_validation import make_validate_keys, verify_frames_aligned


@attr.s
class InMemoryDailyBarReader(CurrencyAwareSessionBarReader):
    """
    A SessionBarReader backed by a dictionary of in-memory DataFrames.

    Parameters
    ----------
    frames : dict[str -> pd.DataFrame]
        Dictionary from field name ("open", "high", "low", "close", or
        "volume") to DataFrame containing data for that field.
    calendar : str or trading_calendars.TradingCalendar
        Calendar (or name of calendar) to which data is aligned.
    currency_codes : pd.Series
        Map from sid -> listing currency for that sid.
    verify_indices : bool, optional
        Whether or not to verify that input data is correctly aligned to the
        given calendar. Default is True.
    """

    _frames = attr.field(
        validator=[attr.validators.instance_of(dict), make_validate_keys(set(OHLCV))]
    )
    _calendar = attr.field(validator=attr.validators.instance_of(TradingCalendar))
    _currency_codes = attr.field(validator=attr.validators.instance_of(pd.Series))
    verify_indices = attr.field(
        default=True, validator=attr.validators.instance_of(bool)
    )
    _sessions = attr.field(init=False)
    _sids = attr.field(init=False)
    _values = attr.field(init=False)

    def __attrs_post_init__(self):
        self._sessions = self._frames["close"].index
        self._sids = self._frames["close"].columns
        self._values = {key: frame.values for key, frame in self._frames.items()}
        if self.verify_indices:
            verify_frames_aligned(list(self._frames.values()), self._calendar)

    @classmethod
    def from_dfs(cls, dfs, calendar, currency_codes):
        """Helper for construction from a dict of DataFrames."""
        return cls(dfs, calendar, currency_codes)

    @property
    def last_available_dt(self):
        return self._calendar[-1]

    @property
    def trading_calendar(self):
        return self._calendar

    @property
    def sessions(self):
        return self._sessions

    def load_raw_arrays(self, columns, start_dt, end_dt, assets):
        if start_dt not in self._sessions:
            raise NoDataOnDate(start_dt)
        if end_dt not in self._sessions:
            raise NoDataOnDate(end_dt)

        asset_indexer = self._sids.get_indexer(assets)
        if -1 in asset_indexer:
            bad_assets = assets[asset_indexer == -1]
            raise NoDataForSid(bad_assets)

        date_indexer = self._sessions.slice_indexer(start_dt, end_dt)

        out = []
        for c in columns:
            out.append(self._values[c][date_indexer, asset_indexer])

        return out

    def get_value(self, sid, dt, field):
        """
        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64-like
            Midnight of the day for which data is requested.
        field : string
            The price field. e.g. ('open', 'high', 'low', 'close', 'volume')

        Returns
        -------
        float
            The spot price for colname of the given sid on the given day.
            Raises a NoDataOnDate exception if the given day and sid is before
            or after the date range of the equity.
            Returns -1 if the day is within the date range, but the price is
            0.
        """
        return self.frames[field].loc[dt, sid]

    def get_last_traded_dt(self, asset, dt):
        """
        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset identifier.
        dt : datetime64-like
            Midnight of the day for which data is requested.

        Returns
        -------
        pd.Timestamp : The last know dt for the asset and dt;
                       NaT if no trade is found before the given dt.
        """
        try:
            return self.frames["close"].loc[:, asset.sid].last_valid_index()
        except IndexError:
            return NaT

    @property
    def first_trading_day(self):
        return self._sessions[0]

    def currency_codes(self, sids):
        codes = self._currency_codes
        return codes.loc[sids].to_numpy()
