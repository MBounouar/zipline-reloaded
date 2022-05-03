"""
HDF5 Pricing File Format
------------------------
At the top level, the file is keyed by exchange (to support files containing multiple exchanges).

Within each exchange, there are 4 subgroups:

``/data``
^^^^^^^^^
Each field (OHLCV) is stored in a dataset as a 2D array, with a row per
sid and a column per session. This differs from the more standard
orientation of dates x sids, because it allows each compressed block to
contain contiguous values for the same sid, which allows for better
compression.

.. code-block:: none

   /data
     /open
     /high
     /low
     /close
     /volume

``/index``
^^^^^^^^^^
Contains two datasets, the index of sids (aligned to the rows of the
OHLCV 2D arrays) and index of sessions (aligned to the columns of the
OHLCV 2D arrays) to use for lookups.

.. code-block:: none

   /index
     /sid
     /day

``/lifetimes``
^^^^^^^^^^^^^^
Contains two datasets, start_date and end_date, defining the lifetime
for each asset, aligned to the sids index.

.. code-block:: none

   /lifetimes
     /start_date
     /end_date

``/currency``
^^^^^^^^^^^^^

Contains a single dataset, ``code``, aligned to the sids index, which contains
the listing currency of each sid.

Example
^^^^^^^
Sample layout of the full file with multiple countries.

.. code-block:: none

   |- /XNYS
   |  |- /data
   |  |  |- /open
   |  |  |- /high
   |  |  |- /low
   |  |  |- /close
   |  |  |- /volume
   |  |
   |  |- /index
   |  |  |- /sid
   |  |  |- /day
   |  |
   |  |- /lifetimes
   |  |  |- /start_date
   |  |  |- /end_date
   |  |
   |  |- /currency
   |  |  |- /code
   |  |
   |  |
   |- /XTSE
      |- /data
      |  |- /open
      |  |- /high
      |  |- /low
      |  |- /close
      |  |- /volume
      |
      |- /index
      |  |- /sid
      |  |- /day
      |
      |- /lifetimes
      |  |- /start_date
      |  |- /end_date
      |
      |- /currency
      |  |- /code
      |
"""

from functools import partial
import os
import h5py
import hdf5plugin
import logbook
import numpy as np
import pandas as pd
from functools import reduce

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
    NoDataForSid,
    NoDataOnDate,
    NotValidDate,
)
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import bytes_array_to_native_str_object_array
from zipline.utils.pandas_utils import check_indexes_all_same
from zipline.utils.calendar_utils import get_calendar, get_calendar_names

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
log = logbook.Logger("HDF5DailyBars")

VERSION = 0

DATA = "data"
INDEX = "index"
LIFETIMES = "lifetimes"
CURRENCY = "currency"
CODE = "code"

SCALING_FACTOR = "scaling_factor"

OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

FIELDS = (OPEN, HIGH, LOW, CLOSE, VOLUME)

DAY = "day"
SID = "sid"

START_DATE = "start_date"
END_DATE = "end_date"

# XXX is reserved for "transactions involving no currency".
MISSING_CURRENCY = "XXX"

DEFAULT_SCALING_FACTORS = {
    # Retain 3 decimal places for prices.
    OPEN: 1000,
    HIGH: 1000,
    LOW: 1000,
    CLOSE: 1000,
    # Volume is expected to be a whole integer.
    VOLUME: 1,
}


def coerce_to_uint32(a, scaling_factor):
    """
    Returns a copy of the array as uint32, applying a scaling factor to
    maintain precision if supplied.
    """
    return (a * scaling_factor).round().astype("uint32")


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


class HDF5OverlappingData(Exception):
    pass


class HDF5BarWriter:
    """
    Class capable of writing daily OHLCV data to disk in a format that
    can be read efficiently by HDF5DailyBarReader.

    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    date_chunk_size : int
        The number of days per chunk in the HDF5 file. If this is
        greater than the number of days in the data, the chunksize will
        match the actual number of days.

    See Also
    --------
    zipline.data.hdf5_daily_bars.HDF5DailyBarReader
    """

    def __init__(self, filename, date_chunk_size, data_frequency="session"):
        self._filename = filename
        self._date_chunk_size = date_chunk_size
        if data_frequency.lower() not in ["minute", "daily", "session"]:
            raise ValueError(
                f"{data_frequency} is not valid only: 'daily', 'session' or 'minute'"
            )
        self._data_frequency = data_frequency

    def h5_file(self, mode):
        return h5py.File(self._filename, mode)

    def _check_valid_calendar_days(self, days):
        if self._data_frequency == "minute":
            is_valid = np.vectorize(self.trading_calendar.is_open_on_minute)
            valid_ts = is_valid(days)
        elif self._data_frequency in ["session", "daily"]:
            is_valid = np.vectorize(self.trading_calendar.is_session)
            valid_ts = is_valid(np.vectorize(partial(pd.Timestamp, tz="UTC"))(days))

        if np.any(~valid_ts):
            raise NotValidDate(f"{days[~valid_ts].astype('datetime64[ns]')}")

    def write(
        self,
        exchange_code,
        frames,
        currency_codes=None,
        scaling_factors=None,
    ):
        """
        Write the OHLCV data for one country to the HDF5 file.

        Parameters
        ----------
        country_code : str
            The ISO 3166 alpha-2 country code for this country.
        frames : dict[str, pd.DataFrame]
            A dict mapping each OHLCV field to a dataframe with a row
            for each date and a column for each sid. The dataframes need
            to have the same index and columns.
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
        if scaling_factors is None:
            scaling_factors = DEFAULT_SCALING_FACTORS

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

        if len(sids):
            chunks = (len(sids), min(self._date_chunk_size, len(days_ns)))
        else:
            # h5py crashes if we provide chunks for empty data.
            chunks = None

        with self.h5_file(mode="a") as self.h5_rwfile:
            # ensure that the file version has been written
            self.h5_rwfile.attrs["version"] = VERSION
            self.h5_rwfile.attrs["data_frequency"] = self._data_frequency

            exchange_group = self.h5_rwfile.get(exchange_code, None)
            if exchange_group is None:
                exchange_group = self.h5_rwfile.create_group(exchange_code)

            self._write_index_group(exchange_group, days_ns, sids)

            self._write_lifetimes_group(exchange_group, sids, start_dates, end_dates)
            self._write_currency_group(exchange_group, currency_codes)
            self._write_data_group(
                exchange_group,
                frames,
                scaling_factors,
                chunks,
            )

    def write_from_sid_df_pairs(
        self,
        exchange_code,
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
                frames={f: empty_frame.copy() for f in FIELDS},
                scaling_factors=scaling_factors,
            )

        self._sids, frames = zip(*data)
        ohlcv_frame = pd.concat(frames)

        # Repeat each sid for each row in its corresponding frame.
        sid_ix = np.repeat(self._sids, [len(f) for f in frames])

        # Add id to the index, so the frame is indexed by (date, id).
        ohlcv_frame.set_index(sid_ix, append=True, inplace=True)

        frames = {field: ohlcv_frame[field].unstack() for field in FIELDS}

        return self.write(
            exchange_code=exchange_code,
            frames=frames,
            scaling_factors=scaling_factors,
            currency_codes=currency_codes,
        )

    def _write_index_group(self, exchange_group, days_ns, sids):
        """Write /country/index."""
        if exchange_group.get(INDEX, None) is None:
            index_group = exchange_group.create_group(INDEX)
            index_group.create_dataset(
                SID,
                data=sids,
                maxshape=(None,),
                dtype="int64",
            )
            index_group.create_dataset(
                DAY,
                data=days_ns,
                maxshape=(None,),
            )
            self._log_writing_dataset(index_group)
        else:
            sid_dataset = exchange_group[INDEX][SID]
            new_sids = np.setdiff1d(sids, sid_dataset[:])
            day_dataset = exchange_group[INDEX][DAY]

            # Check that we don't allow for overwriting sid data
            if np.any(np.isin(sids, sid_dataset[:])) and np.any(
                np.isin(days_ns, day_dataset[:])
            ):
                dts = days_ns[np.isin(days_ns, day_dataset[:])].astype("datetime64[ns]")
                ids = np.intersect1d(sids, sid_dataset[:])
                raise HDF5OverlappingData(
                    f"Data already includes input from date: {dts} and sid: {ids}"
                    "We can only add a new instrument that has same timestamps or append forward"
                )

            if len(days_ns) > 0 and not np.all(np.isin(days_ns, day_dataset[:])):
                day_dataset.resize((day_dataset.shape[0] + days_ns.shape[0]), axis=0)
                day_dataset[-days_ns.shape[0] :] = days_ns

            if len(new_sids) > 0:
                # We have new sids
                sid_dataset.resize((sid_dataset.shape[0] + new_sids.shape[0]), axis=0)
                sid_dataset[-new_sids.shape[0] :] = new_sids

        # h5py does not support datetimes, so they need to be stored
        # as integers.

    def _write_lifetimes_group(self, exchange_group, sids, start_dates, end_dates):
        """Write /country/lifetimes"""
        if exchange_group.get(LIFETIMES, None) is None:
            lifetimes_group = exchange_group.create_group(LIFETIMES)

            if len(start_dates) == 0:
                start_date_ixs = end_date_ixs = np.array([], dtype="int64")
            else:
                start_date_ixs = np.searchsorted(
                    exchange_group[INDEX][DAY][:],
                    start_dates,
                )
                end_date_ixs = np.searchsorted(
                    exchange_group[INDEX][DAY][:],
                    end_dates,
                )

            lifetimes_group.create_dataset(
                START_DATE,
                data=start_date_ixs,
                maxshape=(None,),
            )
            lifetimes_group.create_dataset(
                END_DATE,
                data=end_date_ixs,
                maxshape=(None,),
            )
        else:
            # Do we have new sids or old sids or a mix ?
            start_date_ixs = np.searchsorted(exchange_group[INDEX][DAY][:], start_dates)
            end_date_ixs = np.searchsorted(exchange_group[INDEX][DAY][:], end_dates)

            # We have no previous data just resize and assign the whole data
            if len(exchange_group[LIFETIMES][START_DATE]) == 0:
                exchange_group[LIFETIMES][START_DATE].resize(
                    len(start_date_ixs), axis=0
                )
                exchange_group[LIFETIMES][END_DATE].resize(len(end_date_ixs), axis=0)
                exchange_group[LIFETIMES][START_DATE][:] = start_date_ixs
                exchange_group[LIFETIMES][END_DATE][:] = end_date_ixs
            else:
                new_starts = dict(zip(sids, start_date_ixs))
                new_ends = dict(zip(sids, end_date_ixs))

                sid_dataset = exchange_group[INDEX][SID]
                start_dt_dataset = exchange_group[LIFETIMES][START_DATE]
                end_dt_dataset = exchange_group[LIFETIMES][END_DATE]

                ends_dt = dict(
                    zip(sid_dataset[: len(end_dt_dataset)], end_dt_dataset[:])
                )

                # update the start_date only for new sids
                new_sids = sid_dataset[len(start_dt_dataset) :]
                if len(new_sids):
                    n_start_date_ixs = np.array(
                        [new_starts[sid] for sid in new_sids], dtype="int64"
                    )
                    # Resize according to sids lenght
                    exchange_group[LIFETIMES][START_DATE].resize(
                        len(sid_dataset), axis=0
                    )
                    exchange_group[LIFETIMES][START_DATE][
                        (len(new_sids)) :
                    ] = n_start_date_ixs

                # Resize according to sids lenght
                exchange_group[LIFETIMES][END_DATE].resize(len(sid_dataset), axis=0)

                # Ordering should be preserved
                ends_dt.update(new_ends)

                exchange_group[LIFETIMES][END_DATE][:] = np.array(
                    list(ends_dt), dtype="int64"
                )

        self._log_writing_dataset(exchange_group[LIFETIMES])

    def _write_currency_group(self, exchange_group, currencies):
        """Write /country/currency"""
        if exchange_group.get(CURRENCY, None) is None:
            currency_group = exchange_group.create_group(CURRENCY)
            currency_group.create_dataset(
                CODE,
                data=currencies.values.astype(dtype="S3"),
                maxshape=(None,),
            )
            self._log_writing_dataset(currency_group)
        else:
            currency_group = exchange_group[CURRENCY]
            sid_dataset = exchange_group[INDEX][SID]

            # There is no data just resize and write
            if len(currency_group[CODE]) == 0:
                currency_group[CODE].resize(len(currencies), axis=0)
                currency_group[CODE][:] = currencies.values.astype(dtype="S3")

            # There is a new sid and we update the currency info
            elif len(sid_dataset) > len(currency_group[CODE]):
                start_idx = len(currency_group[CODE])
                new_sids = sid_dataset[start_idx:]
                currency_group[CODE].resize(len(sid_dataset), axis=0)
                currency_group[CODE][start_idx:] = currencies[new_sids].values.astype(
                    dtype="S3"
                )

            self._log_writing_dataset(currency_group)

    def _write_data_group(self, exchange_group, frames, scaling_factors, chunks):
        """Write /country/data"""
        if exchange_group.get(DATA) is None:
            data_group = exchange_group.create_group(DATA)
        else:
            data_group = exchange_group[DATA]

        self._log_writing_dataset(data_group)

        for field in FIELDS:
            frame = frames[field]

            # Sort rows by increasing sid, and columns by increasing date.
            # TODO CHECK SORTING SIDS SOUNDS LIKE A BAD IDEA
            # frame.sort_index(inplace=True) # DISABLED SORTING THIS SOUNDS LIKE A BAD IDEA
            frame.sort_index(axis="columns", inplace=True)

            data = coerce_to_uint32(
                frame.T.fillna(0).values,
                scaling_factors[field],
            )

            if data_group.get(field, None) is None:
                dataset = data_group.create_dataset(
                    field,
                    data=data,
                    maxshape=(None, None),
                    # shuffle=True,
                    # compression="lzf",
                    chunks=True,
                    **hdf5plugin.Blosc(
                        cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
                    ),
                )
                self._log_writing_dataset(dataset)
            else:
                dataset_shape = data_group[field].shape
                if dataset_shape == (0, 0):
                    data_group[field].resize(data.shape)
                    data_group[field][:] = data
                else:
                    nsids = len(exchange_group[INDEX][SID])
                    ndays = len(exchange_group[INDEX][DAY])
                    data_group[field].resize((nsids, ndays))
                    # find which sids need to be inserted and where
                    sid_idx = exchange_group[INDEX][SID][:].searchsorted(
                        self._sids, "left"
                    )
                    data_group[field][sid_idx, -data.shape[1] :] = data

            data_group[field].attrs[SCALING_FACTOR] = scaling_factors[field]

    def _log_writing_dataset(self, dataset):
        log.debug(f"Writing {dataset.name} to file {self._filename}")


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
        [frames[field].isnull().values for field in FIELDS],
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


def convert_with_scaling_factor(a, scaling_factor, to_nan=True):
    # TODO: we should be able to pass the type aware i.e. VOLUME not treated as float
    conversion_factor = 1.0 / scaling_factor
    zeroes = a == 0
    if to_nan:
        return np.where(zeroes, np.nan, a.astype("float64")) * conversion_factor
    else:
        return a * conversion_factor


class HDF5BarReader(CurrencyAwareSessionBarReader):
    """
    Parameters
    ---------
    exchange_group : h5py.Group
        The group for a single country in an HDF5 daily pricing file.
    """

    def __init__(self, exchange_group):
        self._exchange_group = exchange_group

        self._postprocessors = {
            OPEN: partial(
                convert_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(OPEN),
            ),
            HIGH: partial(
                convert_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(HIGH),
            ),
            LOW: partial(
                convert_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(LOW),
            ),
            CLOSE: partial(
                convert_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(CLOSE),
            ),
            # VOLUME: lambda a: a,
            VOLUME: partial(
                convert_with_scaling_factor,
                scaling_factor=self._read_scaling_factor(VOLUME),
                to_nan=False,
            ),
        }

    @classmethod
    def from_file(cls, h5_file, exchange_code):
        """
        Construct from an h5py.File and a country code.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        country_code : str
            The ISO 3166 alpha-2 country code for the country to read.
        """
        if h5_file.attrs["version"] != VERSION:
            raise ValueError(
                f"mismatched version: file is of version {h5_file.attrs['version']},"
                f" expected {VERSION}"
            )
        return cls(h5_file[exchange_code])

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
        return cls.from_file(h5py.File(path, "a"), exchange_code)

    def _read_scaling_factor(self, field):
        return self._exchange_group[DATA][field].attrs[SCALING_FACTOR]

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
        self._validate_timestamp(start_date)
        self._validate_timestamp(end_date)

        start = start_date.asm8
        end = end_date.asm8

        source_date_slice = self._compute_date_range_slice(start, end)

        if self.data_frequency == "minute":
            # All valid session minutes within start_date and end_date
            session_minutes = self.trading_calendar.minutes_in_range(
                start_date, end_date
            )

            n_dates = len(session_minutes)

            dest_date_slice = session_minutes.tz_convert(None).searchsorted(
                self.dates[source_date_slice]
            )

        else:
            dest_date_slice = source_date_slice
            n_dates = source_date_slice.stop - source_date_slice.start

        # Create a buffer into which we'll read data from the h5 file.
        # Allocate an extra row of space that will always contain null values.
        # We'll use that space to provide "data" for entries in ``assets`` that
        # are unknown to us.
        full_buf = np.zeros((len(self.sids) + 1, n_dates), dtype=np.uint32)
        # We'll only read values into this portion of the read buf.
        mutable_buf = full_buf[:-1]

        # Indexer that converts an array aligned to self.sids (which is what we
        # pull from the h5 file) into an array aligned to ``assets``.
        #
        # Unknown assets will have an index of -1, which means they'll always
        # pull from the last row of the read buffer. We allocated an extra
        # empty row above so that these lookups will cause us to fill our
        # output buffer with "null" values.
        sid_selector = self._make_sid_selector(assets)

        out = []
        for column in columns:
            # Zero the buffer to prepare to receive new data.
            mutable_buf.fill(0)
            dataset = self._exchange_group[DATA][column]

            # Fill the mutable portion of our buffer with data from the file.
            if self.data_frequency == "minute":
                dataset.read_direct(
                    mutable_buf,
                    source_sel=np.s_[:, source_date_slice],
                    dest_sel=np.s_[:, dest_date_slice],
                )
            else:
                dataset.read_direct(
                    mutable_buf,
                    np.s_[:, source_date_slice],
                )

            # Select data from the **full buffer**. Unknown assets will pull
            # from the last row, which is always empty.
            out.append(self._postprocessors[column](full_buf[sid_selector].T))

        return out

    def _make_sid_selector(self, assets):
        """
        Build an indexer mapping ``self.sids`` to ``assets``.

        Parameters
        ----------
        assets : list[int]
            List of assets requested by a caller of ``load_raw_arrays``.

        Returns
        -------
        index : np.array[int64]
            Index array containing the index in ``self.sids`` for each location
            in ``assets``. Entries in ``assets`` for which we don't have a sid
            will contain -1. It is caller's responsibility to handle these
            values correctly.
        """
        assets = np.array(assets)
        # sid_selector = self.sids.searchsorted(assets)
        # TODO fix here
        sid_selector = self.sids.searchsorted(assets)
        unknown = np.in1d(assets, self.sids, invert=True)
        sid_selector[unknown] = -1
        return sid_selector

    def _compute_date_range_slice(self, start_date, end_date):
        # Get the index of the start of dates for ``start_date``.
        start_ix = self.dates.searchsorted(start_date)

        # Get the index of the start of the first date **after** end_date.
        end_ix = self.dates.searchsorted(end_date, side="right")

        return slice(start_ix, end_ix)

    def _validate_assets(self, assets):
        """Validate that asset identifiers are contained in the daily bars.

        Parameters
        ----------
        assets : array-like[int]
           The asset identifiers to validate.

        Raises
        ------
        NoDataForSid
            If one or more of the provided asset identifiers are not
            contained in the daily bars.
        """
        missing_sids = np.setdiff1d(assets, self.sids)

        if len(missing_sids):
            raise NoDataForSid(
                f"Assets not contained in daily pricing file: {missing_sids}"
            )

    def _validate_timestamp(self, ts):
        # TODO enforce trading calendar
        if self.trading_calendar is not None:
            if self.data_frequency == "minute":
                if not self.trading_calendar.is_open_on_minute(ts):
                    raise NoDataOnDate(ts)
            elif self.data_frequency in ["daily", "session"]:
                if not self.trading_calendar.is_session(ts):
                    raise NoDataOnDate(ts)
        if ts.asm8 < self.dates[0]:
            raise NoDataBeforeDate(f"requested: {ts} earliest date is: {self.dates[0]}")
        if ts.asm8 > self.dates[-1] and self.data_frequency in ["daily", "session"]:
            raise NoDataAfterDate(f"requested: {ts} latest date is: {self.dates[-1]}")

        # if ts.asm8 > self.dates[-1]:
        #     raise NoDataAfterDate(self.dates[-1])

        elif ts.asm8 not in self.dates:
            return True
        else:
            return False

    @lazyval
    def data_frequency(self):
        return self._exchange_group.parent.attrs["data_frequency"]

    @lazyval
    def version(self):
        return self._exchange_group.parent.attrs["version"]

    @lazyval
    def dates(self):
        return self._exchange_group[INDEX][DAY][:].astype("datetime64[ns]", copy=False)

    @lazyval
    def sids(self):
        return self._exchange_group[INDEX][SID][:]

    @lazyval
    def asset_start_dates(self):
        return self.dates[self._exchange_group[LIFETIMES][START_DATE][:]]

    @lazyval
    def asset_end_dates(self):
        return self.dates[self._exchange_group[LIFETIMES][END_DATE][:]]

    @lazyval
    def _currency_codes(self):
        bytes_array = self._exchange_group[CURRENCY][CODE][:]
        return bytes_array_to_native_str_object_array(bytes_array)

    @lazyval
    def _trading_calendar(self):
        return get_calendar(self._exchange_group.name[1:])

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

    @classmethod
    def cache_clear(cls):
        """ "Method to clear the cache
        After writing to an already instantiated class.
        This is sometimes needed when we are running tests and writing new data.
        We normally don't have to use this in real-cases where data is not continuously written and read
        """
        [v._cache.clear() for _, v in vars(cls).items() if isinstance(v, property)]

    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[object]
            Array of currency codes for listing currencies of ``sids``.
        """
        # Find the index of requested sids in our stored sids.
        # TODO CHECK doesn't work if sids are not sorted and if they are sorted this can become a mess
        # ixs = self.sids.searchsorted(sids, side="left")

        return (
            pd.Series(self._currency_codes, index=self.sids)
            .reindex(sids)
            .replace({np.nan: None})
            .values
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
        """
        self._validate_assets([sid])
        pad_value = self._validate_timestamp(dt)

        # not sorting sids will mess up things
        # TODO CHECK
        # sid_ix = self.sids.searchsorted(sid)
        sid_ix = np.where(self.sids == sid)[0][0]
        dt_ix = self.dates.searchsorted(dt.asm8)

        if pad_value:
            value = np.nan
        else:
            value = self._postprocessors[field](
                self._exchange_group[DATA][field][sid_ix, dt_ix]
            )

        # When the value is nan, this dt may be outside the asset's lifetime.
        # If that's the case, the proper NoDataOnDate exception is raised.
        # Otherwise (when there's just a hole in the middle of the data), the
        # nan is returned.
        if np.isnan(value):
            if dt.asm8 < self.asset_start_dates[sid_ix]:
                raise NoDataBeforeDate(dt)

            if dt.asm8 > self.asset_end_dates[sid_ix]:
                raise NoDataAfterDate(dt)

        return value

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
        # sid_ix = self.sids.searchsorted(asset.sid)
        sid_ix = np.where(self.sids == asset.sid)[0][0]
        # Used to get a slice of all dates up to and including ``dt``.
        dt_limit_ix = self.dates.searchsorted(dt.asm8, side="right")

        # Get the indices of all dates with nonzero volume.
        nonzero_volume_ixs = np.ravel(
            np.nonzero(self._exchange_group[DATA][VOLUME][sid_ix, :dt_limit_ix])
        )

        if len(nonzero_volume_ixs) == 0:
            return pd.NaT

        return pd.Timestamp(self.dates[nonzero_volume_ixs][-1], tz="UTC")


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
        self._exchange_map = pd.concat(
            [
                pd.Series(index=reader.sids, data=exchange_code)
                for exchange_code, reader in readers.items()
            ]
        )

    @classmethod
    def from_file(cls, h5_file):
        """
        Construct from an h5py.File.

        Parameters
        ----------
        h5_file : h5py.File
            An HDF5 daily pricing file.
        """
        return cls(
            {
                exchange_code: HDF5BarReader.from_file(h5_file, exchange_code)
                for exchange_code in h5_file.keys()
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
        return cls.from_file(h5py.File(path, "a"))

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
