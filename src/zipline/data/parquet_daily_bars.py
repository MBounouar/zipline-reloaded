import logging
from functools import partial, reduce

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from zipline.utils.memoize import lazyval


from zipline.data.bar_reader import (
    # NoDataAfterDate,
    # NoDataBeforeDate,
    NoDataForSid,
    # NoDataOnDate,
    NotValidDate,
)
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.pandas_utils import check_indexes_all_same
from zipline.utils.calendar_utils import get_calendar, get_calendar_names
from zipline.utils.data import OHLCV

VERSION = 0
PARQUET_FORMAT_VERSION = "2.6"

# NOTE: XXX is reserved for "transactions involving no currency".
MISSING_CURRENCY = "XXX"

log = logging.getLogger("ParquetDailyBars")


def days_and_sids_for_frames(frames):
    """
    Returns the date index and sid columns shared by a list of dataframes,
    ensuring they all match.

    Parameters
    ----------
    frames : list[pd.DataFrame]
        A list of dataframes indexed by day, with a column per sid.

    Returns
    -------
    days : np.array[datetime64[ns]]
        The days in these dataframes.
    sids : np.array[int64]
        The sids in these dataframes.

    Raises
    ------
    ValueError
        If the dataframes passed are not all indexed by the same days
        and sids.
    """
    if not frames:
        days = np.array([], dtype="datetime64[ns]")
        sids = np.array([], dtype="int64")
        return days, sids

    # Ensure the indices and columns all match.
    check_indexes_all_same(
        [frame.index for frame in frames],
        message="Frames have mismatched days.",
    )
    check_indexes_all_same(
        [frame.columns for frame in frames],
        message="Frames have mismatched sids.",
    )

    return frames[0].index.values.astype("int64"), frames[0].columns.values


def compute_asset_lifetimes(frames):
    """
    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        A dict mapping each OHLCV field to a dataframe with a row for
        each date and a column for each sid, as passed to write().

    Returns
    -------
    start_date_ixs : np.array[int64]
        The index of the first date with non-nan values, for each sid.
    end_date_ixs : np.array[int64]
        The index of the last date with non-nan values, for each sid.
    """
    # Build a 2D array (dates x sids), where an entry is True if all
    # fields are nan for the given day and sid.
    is_null_matrix = np.logical_and.reduce(
        [frames[field].isnull().values for field in OHLCV],
    )
    if not is_null_matrix.size:
        empty = np.array([], dtype="int64")
        return empty, empty.copy()

    # Offset of the first null from the start of the input.
    start_date_ixs = is_null_matrix.argmin(axis=0)
    # Offset of the last null from the **end** of the input.
    end_offsets = is_null_matrix[::-1].argmin(axis=0)
    # Offset of the last null from the start of the input
    end_date_ixs = is_null_matrix.shape[0] - end_offsets - 1

    return start_date_ixs, end_date_ixs


class ParquetBarWriter:
    """Class capable of writing daily OHLCV data to disk in a format that
    can be read efficiently by ParquetDailyBarReader.
    """

    def __init__(self, filename, data_frequency: str = "session") -> None:
        self._filename = filename
        if data_frequency.lower() not in ["minute", "daily", "session"]:
            raise ValueError(
                f"{data_frequency} is not valid only: 'daily', 'session' or 'minute'"
            )
        self._data_frequency = data_frequency

    def _check_valid_calendar_days(self, days):
        if self._data_frequency == "minute":
            is_valid = np.vectorize(self.trading_calendar.is_open_on_minute)
            valid_ts = is_valid(days)
        elif self._data_frequency in ["session", "daily"]:
            is_valid = np.vectorize(self.trading_calendar.is_session)
            valid_ts = is_valid(np.vectorize(partial(pd.Timestamp, tz="UTC"))(days))

        if np.any(~valid_ts):
            raise NotValidDate(f"{days[~valid_ts].astype('datetime64[ns]')}")

    @property
    def progress_bar_message(self):
        return "Merging daily equity files:"

    def progress_bar_item_show_func(self, value):
        return value if value is None else str(value[0])

    def write(
        self,
        exchange_code,
        frames,
        currency_codes=None,
        scaling_factors=None,
    ) -> None:

        # Note that this functions validates that all of the frames
        # share the same days and sids.
        days_ns, sids = days_and_sids_for_frames(list(frames.values()))

        # Check valid exchange_code
        if exchange_code not in get_calendar_names():
            raise ValueError(f"{exchange_code} not a valid exchange code")

        self.trading_calendar = get_calendar(exchange_code)
        # check if days are valid calendar days
        if days_ns.size != 0:
            self._check_valid_calendar_days(days_ns)

        # XXX: We should make this required once we're using it everywhere.
        if currency_codes is None:
            currency_codes = pd.Series(index=sids, data=MISSING_CURRENCY)

        # Currency codes should match dataframe columns.
        check_sids_arrays_match(
            sids,
            currency_codes.index.values,
            message="currency_codes sids do not match data sids:",
        )

        # Write start and end dates for each sid.
        start_date_ixs, end_date_ixs = compute_asset_lifetimes(frames)
        start_dates = days_ns[start_date_ixs]
        end_dates = days_ns[end_date_ixs]

    def write_from_sid_df_pairs(
        self,
        exchange_code: str,
        data,
        currency_codes=None,
        scaling_factors=None,
    ):
        """
        Parameters
        ----------
        exchange_code : str
            The ISO-10383 market exchange code.
        data : iterable[tuple[int, pandas.DataFrame]]
            The data chunks to write. Each chunk should be a tuple of
            sid and the data for that asset.
        currency_codes : pd.Series, optional
            Series mapping sids to 3-digit currency code values for those sids'
            listing currencies. If not passed, missing currencies will be
            written.
        scaling_factors : dict[str, float], optional
            A dict mapping each OHLCV field to a scaling factor, which
            is applied (as a multiplier) to the values of field to
            efficiently store them as uint32, while maintaining desired
            precision. These factors are written to the file as metadata,
            which is consumed by the reader to adjust back to the original
            float values. Default is None, in which case
            DEFAULT_SCALING_FACTORS is used.
        """
        data = list(data)
        if not data:
            empty_frame = pd.DataFrame(
                data=None,
                index=np.array([], dtype="datetime64[ns]"),
                columns=np.array([], dtype="int64"),
            )
            return self.write(
                exchange_code=exchange_code,
                frames={f: empty_frame.copy() for f in OHLCV},
                scaling_factors=scaling_factors,
            )

        self._sids, frames = zip(*data)
        ohlcv_frame = pd.concat(frames)

        # Repeat each sid for each row in its corresponding frame.
        sid_ix = np.repeat(self._sids, [len(f) for f in frames])

        # Add id to the index, so the frame is indexed by (date, id).
        ohlcv_frame.set_index(sid_ix, append=True, inplace=True)

        frames = {field: ohlcv_frame[field].unstack() for field in OHLCV}

        return self.write(
            exchange_code=exchange_code,
            frames=frames,
            scaling_factors=scaling_factors,
            currency_codes=currency_codes,
        )


class ParquetBarReader(CurrencyAwareSessionBarReader):
    """_summary_

    Parameters
    ----------
    CurrencyAwareSessionBarReader : _type_
        _description_
    """

    def __init__(self, exchange_group):
        self._exchange_group = exchange_group

    @classmethod
    def from_file(cls, pq_file, exchange_code):
        """
        Construct from an parquet.File and an exchange code.

        Parameters
        ----------

        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        if pq_file.attrs["version"] != VERSION:
            raise ValueError(
                f"mismatched version: file is of version {pq_file.attrs['version']},"
                f" expected {VERSION}"
            )
        return cls(pq_file[exchange_code])

    @classmethod
    def from_path(cls, path, exchange_code):
        """
        Construct from a file path and a country code.

        Parameters
        ----------
        path : str
            The path to an HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        # return cls.from_file(h5py.File(path, "a"), exchange_code)
        ...

    def _read_scaling_factor(self, field):
        ...

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        ...

    @lazyval
    def data_frequency(self):
        return self._exchange_group.parent.attrs["data_frequency"]

    @lazyval
    def version(self):
        return self._exchange_group.parent.attrs["version"]

    @lazyval
    def dates(self):
        ...
        # return self._exchange_group[INDEX][DAY][:].astype("datetime64[ns]", copy=False)

    @lazyval
    def sids(self):
        ...
        # return self._exchange_group[INDEX][SID][:]

    @lazyval
    def asset_start_dates(self):
        ...
        # return self.dates[self._exchange_group[LIFETIMES][START_DATE][:]]

    @lazyval
    def asset_end_dates(self):
        ...
        # return self.dates[self._exchange_group[LIFETIMES][END_DATE][:]]

    @lazyval
    def _currency_codes(self):
        ...
        # bytes_array = self._exchange_group[CURRENCY][CODE][:]
        # return bytes_array_to_native_str_object_array(bytes_array)

    @lazyval
    def _trading_calendar(self):
        ...
        # return get_calendar(self._exchange_group.name[1:])

    @lazyval
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return pd.Timestamp(self.dates[-1], tz="UTC")

    @lazyval
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        return self._trading_calendar

    @lazyval
    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return pd.Timestamp(self.dates[0], tz="UTC")

    @lazyval
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(self.dates, utc=True)


class MultiExchangeDailyBarReader(CurrencyAwareSessionBarReader):
    """

    Parameters
    ---------
    readers : dict[str -> SessionBarReader]
        A dict mapping country codes to SessionBarReader instances to
        service each country.
    """

    def __init__(self, readers):
        self._readers = readers
        self._country_map = pd.concat(
            [
                pd.Series(index=reader.sids, data=country_code)
                for country_code, reader in readers.items()
            ]
        )

    @classmethod
    def from_file(cls, pq_file):

        return cls(
            {
                exchange_code: None  # P.from_file(h5_file, exchange_code)
                for exchange_code in []  # h5_file.keys()
            }
        )

    @classmethod
    def from_path(cls, path):
        """
        Construct from a file path.

        Parameters
        ----------
        path : str
            Path to an HDF5 daily pricing file.
        """
        return cls.from_file(path)

    @property
    def exchanges(self):
        """A set-like object of the exchange codes supplied by this reader."""
        return self._readers.keys()

    def _exchange_code_for_assets(self, assets):
        exchange_codes = self._exchange_map.reindex(assets)

        # Series.get() returns None if none of the labels are in the index.
        if exchange_codes is not None:
            unique_exchange_codes = exchange_codes.dropna().unique()
            num_exchanges = len(unique_exchange_codes)
        else:
            num_exchanges = 0

        if num_exchanges == 0:
            raise ValueError("At least one valid asset id is required.")
        elif num_exchanges > 1:
            raise NotImplementedError(
                f"Assets were requested from multiple exchanges ({list(unique_exchange_codes)}),"
                " but multi-country reads are not yet supported."
            )

        return unique_exchange_codes.item()

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """
        Parameters
        ----------
        columns : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_date: Timestamp
           Beginning of the window range.
        end_date: Timestamp
           End of the window range.
        assets : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (minutes in range, sids) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        country_code = self._exchange_code_for_assets(assets)

        return self._readers[country_code].load_raw_arrays(
            columns,
            start_date,
            end_date,
            assets,
        )

    @property
    def last_available_dt(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The last session for which the reader can provide data.
        """
        return max(reader.last_available_dt for reader in self._readers.values())

    @property
    def trading_calendar(self):
        """
        Returns the zipline.utils.calendar.trading_calendar used to read
        the data.  Can be None (if the writer didn't specify it).
        """
        return [reader.trading_calendar for reader in self._readers.values()]

    @property
    def first_trading_day(self):
        """
        Returns
        -------
        dt : pd.Timestamp
            The first trading day (session) for which the reader can provide
            data.
        """
        return min(reader.first_trading_day for reader in self._readers.values())

    @property
    def sessions(self):
        """
        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """
        return pd.to_datetime(
            reduce(
                np.union1d,
                (reader.dates for reader in self._readers.values()),
            ),
            utc=True,
        )

    def get_value(self, sid, dt, field):
        """
        Retrieve the value at the given coordinates.

        Parameters
        ----------
        sid : int
            The asset identifier.
        dt : pd.Timestamp
            The timestamp for the desired data point.
        field : string
            The OHLVC name for the desired data point.

        Returns
        -------
        value : float|int
            The value at the given coordinates, ``float`` for OHLC, ``int``
            for 'volume'.

        Raises
        ------
        NoDataOnDate
            If the given dt is not a valid market minute (in minute mode) or
            session (in daily mode) according to this reader's tradingcalendar.
        NoDataForSid
            If the given sid is not valid.
        """
        try:
            exchange_code = self._exchange_code_for_assets([sid])
        except ValueError as exc:
            raise NoDataForSid(
                f"Asset not contained in daily pricing file: {sid}"
            ) from exc
        return self._readers[exchange_code].get_value(sid, dt, field)

    def get_last_traded_dt(self, asset, dt):
        """
        Get the latest day on or before ``dt`` in which ``asset`` traded.

        If there are no trades on or before ``dt``, returns ``pd.NaT``.

        Parameters
        ----------
        asset : zipline.asset.Asset
            The asset for which to get the last traded day.
        dt : pd.Timestamp
            The dt at which to start searching for the last traded day.

        Returns
        -------
        last_traded : pd.Timestamp
            The day of the last trade for the given asset, using the
            input dt as a vantage point.
        """
        exchange_code = self._exchange_code_for_assets([asset.sid])
        return self._readers[exchange_code].get_last_traded_dt(asset, dt)

    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Assumes that a sid's prices are always quoted in a single currency.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[S3]
            Array of currency codes for listing currencies of ``sids``.
        """
        exchange_code = self._exchange_code_for_assets(sids)
        return self._readers[exchange_code].currency_codes(sids)


def check_sids_arrays_match(left, right, message):
    """Check that two 1d arrays of sids are equal"""
    if len(left) != len(right):
        raise ValueError(
            f"{message}:\nlen(left) ({len(left)}) != len(right) ({len(right)})"
        )

    diff = left != right
    if diff.any():
        (bad_locs,) = np.where(diff)
        raise ValueError(f"{message}:\n Indices with differences: {bad_locs}")
