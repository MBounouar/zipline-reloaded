#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from datetime import timedelta
from unittest import skip

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from zipline.data.bar_reader import NoDataForSid, NoDataOnDate, NotValidDate
from zipline.data.hdf5_daily_bars import (
    HDF5OverlappingData,
    HDF5BarWriter,
    HDF5BarReader,
    VERSION as HDF5_FILE_VERSION,
)
from zipline.data.bcolz_minute_bars import (
    US_EQUITIES_MINUTES_PER_DAY,
    BcolzMinuteBarMetadata,
    # BcolzMinuteBarReader,
    # BcolzMinuteBarWriter,
    BcolzMinuteOverlappingData,
    BcolzMinuteWriterColumnMismatch,
    H5MinuteBarUpdateReader,
    H5MinuteBarUpdateWriter,
)
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithInstanceTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
)

# Calendar is set to cover several half days, to check a case where half
# days would be read out of order in cases of windows which spanned over
# multiple half days.
TEST_CALENDAR_START = pd.Timestamp("2014-06-02", tz="UTC")
TEST_CALENDAR_STOP = pd.Timestamp("2015-12-31", tz="UTC")


class HDF5MinuteBarTestCase(
    WithTradingCalendars,
    WithAssetFinder,
    WithInstanceTmpDir,
    ZiplineTestCase,
):
    ASSET_FINDER_EQUITY_SIDS = 1, 2

    @classmethod
    def init_class_fixtures(cls):
        super(HDF5MinuteBarTestCase, cls).init_class_fixtures()

        cal = cls.trading_calendar.schedule.loc[TEST_CALENDAR_START:TEST_CALENDAR_STOP]

        cls.market_opens = cal.market_open.dt.tz_localize("UTC")
        cls.market_closes = cal.market_close.dt.tz_localize("UTC")

        cls.test_calendar_start = cls.market_opens.index[0]
        cls.test_calendar_stop = cls.market_opens.index[-1]

    def init_instance_fixtures(self):
        super(HDF5MinuteBarTestCase, self).init_instance_fixtures()

        self.dest = self.instance_tmpdir.getpath("")
        self.file_path = Path(self.dest, "minute_bars.h5")
        self.writer = HDF5BarWriter(
            self.file_path,
            US_EQUITIES_MINUTES_PER_DAY,
            data_frequency="minute",
        )
        self.writer.write_from_sid_df_pairs(
            "US", iter(()), exchange_name=self.trading_calendar.name
        )
        self.reader = HDF5BarReader.from_path(self.file_path, "US")

    def test_version(self):
        hdf5_version = self.reader.version
        assert hdf5_version == HDF5_FILE_VERSION

    def test_no_minute_bars_for_sid(self):
        minute = self.market_opens[self.test_calendar_start]
        with pytest.raises(NoDataForSid):
            self.reader.get_value(1337, minute, "close")

        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
                "id": [sid],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        for field in ["open", "high", "low", "close", "volume"]:
            val = self.reader.get_value(sid, minute, field)
            assert val == data.loc[minute, field]

    def test_precision_after_scaling(self):
        """For numbers that don't have an exact float representation,
        assert that scaling the value does not cause a loss in precision.
        """
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [130.23],
                "high": [130.23],
                "low": [130.23],
                "close": [130.23],
                "volume": [1000],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))
        for field in data:
            val = self.reader.get_value(sid, minute, field)
            assert val == data.loc[minute, field]

    def test_write_one_ohlcv_with_ratios(self):
        # TODO That's not the really working as intended
        # scaling factors are not overwritten apparently
        # is not SID specific
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        file_name = self.writer._filename
        writer = HDF5BarWriter(
            file_name, US_EQUITIES_MINUTES_PER_DAY, data_frequency="minute"
        )
        # Create a new writer with `ohlc_ratios_per_sid` defined.
        scaling_factors = dict(zip(data.columns, [25, 25, 25, 25, 10]))
        writer.write_from_sid_df_pairs(
            "US",
            ((sid, data),),
            scaling_factors=scaling_factors,
        )

        reader = HDF5BarReader.from_path(file_name, "US")

        for field in data:
            val = reader.get_value(sid, minute, field)
            assert val == data.loc[minute, field]
            assert (
                reader._country_group["data"][field].attrs["scaling_factor"]
                == scaling_factors[field]
            )

    def test_write_two_bars(self):
        minute_0 = self.market_opens[self.test_calendar_start]
        minute_1 = minute_0 + timedelta(minutes=1)
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0, 11.0],
                "high": [20.0, 21.0],
                "low": [30.0, 31.0],
                "close": [40.0, 41.0],
                "volume": [50, 51],
            },
            index=[minute_0, minute_1],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        for minute_i in data.index:
            for k, v in data.loc[minute_i].items():
                assert v == self.reader.get_value(sid, minute_i, k)

    def test_write_on_second_day(self):
        second_day = self.test_calendar_start + timedelta(days=1)
        minute = self.market_opens[second_day]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))
        for field in data:
            val = self.reader.get_value(sid, minute, field)
            assert val == data.loc[minute, field]

    def test_write_empty(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [0],
                "high": [0],
                "low": [0],
                "close": [0],
                "volume": [0],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        for field in data:
            val = self.reader.get_value(sid, minute, field)
            if field == "volume":
                assert val == 0
            else:
                assert np.isnan(val)

    def test_write_on_multiple_days(self):
        tds = self.market_opens.index
        days = tds[
            tds.slice_indexer(
                start=self.test_calendar_start + timedelta(days=1),
                end=self.test_calendar_start + timedelta(days=3),
            )
        ]
        minutes = pd.DatetimeIndex(
            [
                self.market_opens[days[0]] + timedelta(minutes=60),
                self.market_opens[days[1]] + timedelta(minutes=120),
            ]
        )
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0, 11.0],
                "high": [20.0, 21.0],
                "low": [30.0, 31.0],
                "close": [40.0, 41.0],
                "volume": [50.0, 51.0],
            },
            index=minutes,
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        minute = minutes[0]

        for field in data:
            for minute in minutes:
                val = self.reader.get_value(sid, minute, field)
                assert val == data.loc[minute, field]

    def test_no_overwrite(self):
        # We don't want to overwrite timestamp data already present
        #
        minute = self.market_opens[TEST_CALENDAR_START]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        with pytest.raises(HDF5OverlappingData):
            self.writer.write_from_sid_df_pairs("US", ((sid, data),))

    def test_append_to_same_day(self):
        """Test writing data with the same date as existing data in our file."""
        sid = 1
        first_minute = self.market_opens[TEST_CALENDAR_START]
        data = pd.DataFrame(
            data={
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[first_minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        # Write data in the same day as the previous minute
        second_minute = first_minute + pd.Timedelta(minutes=1)
        new_data = pd.DataFrame(
            data={
                "open": [5.0],
                "high": [10.0],
                "low": [3.0],
                "close": [7.0],
                "volume": [10.0],
            },
            index=[second_minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, new_data),))

        for field in new_data:
            val = self.reader.get_value(sid, second_minute, field)
            assert new_data.loc[second_minute, field] == val

    def test_append_on_new_day(self):
        sid = 1

        ohlcv = {
            "open": [2.0],
            "high": [3.0],
            "low": [1.0],
            "close": [2.0],
            "volume": [10.0],
        }

        dt = self.market_opens[TEST_CALENDAR_STOP]
        data = pd.DataFrame(data=ohlcv, index=[dt])
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))
        file_name = self.writer._filename
        # Open a new writer to cover `open` method, also a common usage
        # of appending new days will be writing to an existing directory.
        cday = self.trading_calendar.schedule.index.freq
        # new_end_session = TEST_CALENDAR_STOP + cday
        writer = HDF5BarWriter(
            file_name, US_EQUITIES_MINUTES_PER_DAY, data_frequency="minute"
        )
        next_day_minute = dt + cday
        new_data = pd.DataFrame(data=ohlcv, index=[next_day_minute])
        writer.write_from_sid_df_pairs("US", ((sid, new_data * 0.5),))

        # Get a new reader to test updated calendar.
        reader = HDF5BarReader.from_path(file_name, "US")

        second_minute = dt + pd.Timedelta(minutes=1)

        # The second minute should have been padded with zeros
        # TODO: Why pad with zeros ? No data is no data ! ffill maybe ?
        for col in ("open", "high", "low", "close", "volume"):
            assert_almost_equal(np.nan, reader.get_value(sid, second_minute, col))
        # assert 0 == reader.get_value(sid, second_minute, "volume")

        # The next day minute should have data.
        for col in ("open", "high", "low", "close", "volume"):
            assert_almost_equal(
                (new_data * 0.5).loc[next_day_minute, col],
                reader.get_value(sid, next_day_minute, col),
            )

    def test_write_multiple_sids(self):
        """Test writing multiple sids.

        Tests both that the data is written to the correct sid, as well as
        ensuring that the logic for creating the subdirectory path to each sid
        does not cause issues from attempts to recreate existing paths.
        (Calling out this coverage, because an assertion of that logic does not
        show up in the test itself, but is exercised by the act of attempting
        to write two consecutive sids, which would be written to the same
        containing directory, `00/00/000001.bcolz` and `00/00/000002.bcolz)

        Before applying a check to make sure the path writing did not
        re-attempt directory creation an OSError like the following would
        occur:

        ```
        OSError: [Errno 17] File exists: '/tmp/tmpR7yzzT/minute_bars/00/00'
        ```
        """
        minute = self.market_opens[TEST_CALENDAR_START]
        sids = [1, 2]
        data_1 = pd.DataFrame(
            data={
                "open": [15.0],
                "high": [17.0],
                "low": [11.0],
                "close": [15.0],
                "volume": [100.0],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sids[0], data_1),))

        data_2 = pd.DataFrame(
            data={
                "open": [25.0],
                "high": [27.0],
                "low": [21.0],
                "close": [25.0],
                "volume": [200.0],
            },
            index=[minute],
        )
        self.writer.write_from_sid_df_pairs("US", ((sids[1], data_2),))

        for sid, data in zip(sids, [data_1, data_2]):
            for field in data:
                assert data.loc[minute, field] == self.reader.get_value(
                    sid, minute, field
                )

    def test_pad_data(self):
        """Test writing empty data."""
        sid = 1
        last_date = self.writer.last_date_in_output_for_sid(sid)
        assert last_date is pd.NaT

        self.writer.pad(sid, TEST_CALENDAR_START)

        last_date = self.writer.last_date_in_output_for_sid(sid)
        assert last_date == TEST_CALENDAR_START

        freq = self.market_opens.index.freq
        day = TEST_CALENDAR_START + freq
        minute = self.market_opens[day]

        data = pd.DataFrame(
            data={
                "open": [15.0],
                "high": [17.0],
                "low": [11.0],
                "close": [15.0],
                "volume": [100.0],
            },
            index=[minute],
        )
        self.writer.write_sid(sid, data)

        open_price = self.reader.get_value(sid, minute, "open")
        assert 15.0 == open_price

        high_price = self.reader.get_value(sid, minute, "high")
        assert 17.0 == high_price

        low_price = self.reader.get_value(sid, minute, "low")
        assert 11.0 == low_price

        close_price = self.reader.get_value(sid, minute, "close")
        assert 15.0 == close_price

        volume_price = self.reader.get_value(sid, minute, "volume")
        assert 100.0 == volume_price

        # Check that if we then pad the rest of this day, we end up with
        # 2 days worth of minutes.
        self.writer.pad(sid, day)

        assert len(self.writer._ensure_ctable(sid)) == self.writer._minutes_per_day * 2

    def test_nans(self):
        """Test writing empty data."""
        sid = 1
        last_date = self.writer.last_date_in_output_for_sid(sid)
        assert last_date is pd.NaT

        self.writer.pad(sid, TEST_CALENDAR_START)

        last_date = self.writer.last_date_in_output_for_sid(sid)
        assert last_date == TEST_CALENDAR_START

        freq = self.market_opens.index.freq
        minute = self.market_opens[TEST_CALENDAR_START + freq]
        minutes = pd.date_range(minute, periods=9, freq="min")
        data = pd.DataFrame(
            data={
                "open": np.full(9, np.nan),
                "high": np.full(9, np.nan),
                "low": np.full(9, np.nan),
                "close": np.full(9, np.nan),
                "volume": np.full(9, 0.0),
            },
            index=minutes,
        )
        self.writer.write_sid(sid, data)

        fields = ["open", "high", "low", "close", "volume"]

        ohlcv_window = list(
            map(
                np.transpose,
                self.reader.load_raw_arrays(
                    fields,
                    minutes[0],
                    minutes[-1],
                    [sid],
                ),
            )
        )

        for i, field in enumerate(fields):
            if field != "volume":
                assert_array_equal(np.full(9, np.nan), ohlcv_window[i][0])
            else:
                assert_array_equal(np.zeros(9), ohlcv_window[i][0])

    def test_differing_nans(self):
        """Also test nans of differing values/construction."""
        sid = 1
        last_date = self.writer.last_date_in_output_for_sid(sid)
        assert last_date is pd.NaT

        self.writer.pad(sid, TEST_CALENDAR_START)

        last_date = self.writer.last_date_in_output_for_sid(sid)
        assert last_date == TEST_CALENDAR_START

        freq = self.market_opens.index.freq
        minute = self.market_opens[TEST_CALENDAR_START + freq]
        minutes = pd.date_range(minute, periods=9, freq="min")
        data = pd.DataFrame(
            data={
                "open": ((0b11111111111 << 52) + np.arange(1, 10, dtype=np.int64)).view(
                    np.float64
                ),
                "high": (
                    (0b11111111111 << 52) + np.arange(11, 20, dtype=np.int64)
                ).view(np.float64),
                "low": ((0b11111111111 << 52) + np.arange(21, 30, dtype=np.int64)).view(
                    np.float64
                ),
                "close": (
                    (0b11111111111 << 52) + np.arange(31, 40, dtype=np.int64)
                ).view(np.float64),
                "volume": np.full(9, 0.0),
            },
            index=minutes,
        )
        self.writer.write_sid(sid, data)

        fields = ["open", "high", "low", "close", "volume"]

        ohlcv_window = list(
            map(
                np.transpose,
                self.reader.load_raw_arrays(
                    fields,
                    minutes[0],
                    minutes[-1],
                    [sid],
                ),
            )
        )

        for i, field in enumerate(fields):
            if field != "volume":
                assert_array_equal(np.full(9, np.nan), ohlcv_window[i][0])
            else:
                assert_array_equal(np.zeros(9), ohlcv_window[i][0])

    def test_write_cols(self):
        minute_0 = self.market_opens[self.test_calendar_start].tz_localize(None)
        minute_1 = minute_0 + timedelta(minutes=1)
        sid = 1
        cols = {
            "open": np.array([10.0, 11.0]),
            "high": np.array([20.0, 21.0]),
            "low": np.array([30.0, 31.0]),
            "close": np.array([40.0, 41.0]),
            "volume": np.array([50.0, 51.0]),
        }
        dts = np.array([minute_0, minute_1], dtype="datetime64[s]")
        self.writer.write_cols(sid, dts, cols)

        open_price = self.reader.get_value(sid, minute_0, "open")
        assert 10.0 == open_price

        high_price = self.reader.get_value(sid, minute_0, "high")
        assert 20.0 == high_price

        low_price = self.reader.get_value(sid, minute_0, "low")
        assert 30.0 == low_price

        close_price = self.reader.get_value(sid, minute_0, "close")
        assert 40.0 == close_price

        volume_price = self.reader.get_value(sid, minute_0, "volume")
        assert 50.0 == volume_price

        open_price = self.reader.get_value(sid, minute_1, "open")
        assert 11.0 == open_price

        high_price = self.reader.get_value(sid, minute_1, "high")
        assert 21.0 == high_price

        low_price = self.reader.get_value(sid, minute_1, "low")
        assert 31.0 == low_price

        close_price = self.reader.get_value(sid, minute_1, "close")
        assert 41.0 == close_price

        volume_price = self.reader.get_value(sid, minute_1, "volume")
        assert 51.0 == volume_price

    def test_write_cols_mismatch_length(self):
        dts = pd.date_range(
            self.market_opens[self.test_calendar_start].tz_localize(None),
            periods=2,
            freq="min",
        ).asi8.astype("datetime64[s]")
        sid = 1
        cols = {
            "open": np.array([10.0, 11.0, 12.0]),
            "high": np.array([20.0, 21.0]),
            "low": np.array([30.0, 31.0, 33.0, 34.0]),
            "close": np.array([40.0, 41.0]),
            "volume": np.array([50.0, 51.0, 52.0]),
        }
        with pytest.raises(BcolzMinuteWriterColumnMismatch):
            self.writer.write_cols(sid, dts, cols)

    def test_unadjusted_minutes(self):
        """Test unadjusted minutes"""

        start_minute = self.market_opens[TEST_CALENDAR_START]
        minutes = [
            start_minute,
            start_minute + pd.Timedelta("1 min"),
            start_minute + pd.Timedelta("2 min"),
        ]
        sids = [1, 2]
        data_1 = pd.DataFrame(
            data={
                "open": [15.0, np.nan, 15.1],
                "high": [17.0, np.nan, 17.1],
                "low": [11.0, np.nan, 11.1],
                "close": [14.0, np.nan, 14.1],
                "volume": [1000, 0, 1001],
            },
            index=minutes,
        )
        self.writer.write_from_sid_df_pairs(
            "US",
            ((sids[0], data_1),),
            exchange_name=self.trading_calendar.name,
        )

        data_2 = pd.DataFrame(
            data={
                "open": [25.0, np.nan, 25.1],
                "high": [27.0, np.nan, 27.1],
                "low": [21.0, np.nan, 21.1],
                "close": [24.0, np.nan, 24.1],
                "volume": [2000, 0, 2001],
            },
            index=minutes,
        )

        self.writer.write_from_sid_df_pairs(
            "US",
            ((sids[1], data_2),),
            exchange_name=self.trading_calendar.name,
        )

        file_name = self.writer._filename
        reader = HDF5BarReader.from_path(file_name, "US")

        columns = ["open", "high", "low", "close", "volume"]

        arrays = list(
            map(
                np.transpose,
                reader.load_raw_arrays(
                    columns,
                    minutes[0],
                    minutes[-1],
                    sids,
                ),
            )
        )

        data = {sids[0]: data_1, sids[1]: data_2}

        for i, col in enumerate(columns):
            for j, sid in enumerate(sids):
                assert_almost_equal(data[sid][col], arrays[i][j])

    def test_unadjusted_minutes_early_close(self):
        """Test unadjusted minute window,
        ensuring that early closes are filtered out.
        """
        # TODO, Check the logic if we really want this and test for it correctly
        day_before_thanksgiving = pd.Timestamp("2015-11-25", tz="UTC")
        xmas_eve = pd.Timestamp("2015-12-24", tz="UTC")
        market_day_after_xmas = pd.Timestamp("2015-12-28", tz="UTC")

        minutes = [
            self.market_closes[day_before_thanksgiving] - pd.Timedelta("2 min"),
            self.market_closes[xmas_eve] - pd.Timedelta("1 min"),
            self.market_opens[market_day_after_xmas] + pd.Timedelta("1 min"),
        ]
        sids = [1, 2]
        data_1 = pd.DataFrame(
            data={
                "open": [15.0, 15.1, 15.2],
                "high": [17.0, 17.1, 17.2],
                "low": [11.0, 11.1, 11.3],
                "close": [14.0, 14.1, 14.2],
                "volume": [1000, 1001, 1002],
            },
            index=minutes,
        )
        self.writer.write_from_sid_df_pairs(
            "US",
            ((sids[0], data_1),),
            exchange_name=self.trading_calendar.name,
        )

        data_2 = pd.DataFrame(
            data={
                "open": [25.0, 25.1, 25.2],
                "high": [27.0, 27.1, 27.2],
                "low": [21.0, 21.1, 21.2],
                "close": [24.0, 24.1, 24.2],
                "volume": [2000, 2001, 2002],
            },
            index=minutes,
        )
        self.writer.write_from_sid_df_pairs(
            "US",
            ((sids[1], data_2),),
            exchange_name=self.trading_calendar.name,
        )

        reader = self.reader

        columns = ["open", "high", "low", "close", "volume"]

        arrays = list(
            map(
                np.transpose,
                reader.load_raw_arrays(
                    columns,
                    minutes[0],
                    minutes[-1],
                    sids,
                ),
            )
        )

        data = {sids[0]: data_1, sids[1]: data_2}

        start_minute_loc = self.trading_calendar.all_minutes.get_loc(minutes[0])
        minute_locs = [
            self.trading_calendar.all_minutes.get_loc(minute) - start_minute_loc
            for minute in minutes
        ]

        for i, col in enumerate(columns):
            for j, sid in enumerate(sids):
                assert_almost_equal(
                    data[sid].loc[minutes, col], arrays[i][j]  # [minute_locs]
                )

    def test_adjust_non_trading_minutes(self):
        start_day = pd.Timestamp("2015-06-01", tz="UTC")
        end_day = pd.Timestamp("2015-06-02", tz="UTC")

        sid = 1
        cols = {
            "open": np.arange(1, 781),
            "high": np.arange(1, 781),
            "low": np.arange(1, 781),
            "close": np.arange(1, 781),
            "volume": np.arange(1, 781),
        }
        dts = np.array(
            self.trading_calendar.minutes_for_sessions_in_range(
                self.trading_calendar.minute_to_session_label(start_day),
                self.trading_calendar.minute_to_session_label(end_day),
            )
        )
        data = pd.DataFrame(cols, index=dts)
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        assert (
            self.reader.get_value(
                sid, pd.Timestamp("2015-06-01 20:00:00", tz="UTC"), "open"
            )
            == 390
        )
        assert (
            self.reader.get_value(
                sid, pd.Timestamp("2015-06-02 20:00:00", tz="UTC"), "open"
            )
            == 780
        )

        with pytest.raises(NoDataOnDate):
            self.reader.get_value(sid, pd.Timestamp("2015-06-02", tz="UTC"), "open")

        with pytest.raises(NoDataOnDate):
            self.reader.get_value(
                sid, pd.Timestamp("2015-06-02 20:01:00", tz="UTC"), "open"
            )

    def test_adjust_non_trading_minutes_half_days(self):
        # TODO CHECK THE BEHAVIOUR
        # half day
        start_day = pd.Timestamp("2015-11-27", tz="UTC")
        end_day = pd.Timestamp("2015-11-30", tz="UTC")

        sid = 1
        cols = {
            "open": np.arange(1, 601),
            "high": np.arange(1, 601),
            "low": np.arange(1, 601),
            "close": np.arange(1, 601),
            "volume": np.arange(1, 601),
        }
        dts = np.array(
            self.trading_calendar.minutes_for_sessions_in_range(
                self.trading_calendar.minute_to_session_label(start_day),
                self.trading_calendar.minute_to_session_label(end_day),
            )
        )

        data = pd.DataFrame(cols, index=dts)
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        assert (
            self.reader.get_value(
                sid, pd.Timestamp("2015-11-27 18:00:00", tz="UTC"), "open"
            )
            == 210
        )
        assert (
            self.reader.get_value(
                sid, pd.Timestamp("2015-11-30 21:00:00", tz="UTC"), "open"
            )
            == 600
        )

        assert (
            self.reader.get_value(
                sid, pd.Timestamp("2015-11-27 18:01:00", tz="UTC"), "open"
            )
            == 210
        )

        with pytest.raises(NoDataOnDate):
            self.reader.get_value(sid, pd.Timestamp("2015-11-30", tz="UTC"), "open")

        with pytest.raises(NoDataOnDate):
            self.reader.get_value(
                sid, pd.Timestamp("2015-11-30 21:01:00", tz="UTC"), "open"
            )

    def test_set_sid_attrs(self):
        """Confirm that we can set the attributes of a sid's file correctly."""

        sid = 1
        start_day = pd.Timestamp("2015-11-27", tz="UTC")
        end_day = pd.Timestamp("2015-06-02", tz="UTC")
        attrs = {
            "start_day": start_day.value / int(1e9),
            "end_day": end_day.value / int(1e9),
            "factor": 100,
        }

        # Write the attributes
        self.writer.set_sid_attrs(sid, **attrs)
        # Read the attributes
        for k, v in attrs.items():
            assert self.reader.get_sid_attr(sid, k) == v

    def test_truncate_between_data_points(self):
        tds = self.market_opens.index
        days = tds[
            tds.slice_indexer(
                start=self.test_calendar_start + timedelta(days=1),
                end=self.test_calendar_start + timedelta(days=3),
            )
        ]
        minutes = pd.DatetimeIndex(
            [
                self.market_opens[days[0]] + timedelta(minutes=60),
                self.market_opens[days[1]] + timedelta(minutes=120),
            ]
        )
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0, 11.0],
                "high": [20.0, 21.0],
                "low": [30.0, 31.0],
                "close": [40.0, 41.0],
                "volume": [50.0, 51.0],
            },
            index=minutes,
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        # Open a new writer to cover `open` method, also truncating only
        # applies to an existing directory.
        # writer = self.writer

        # # Truncate to first day with data.
        # writer.truncate(days[0])

        # # Refresh the reader since truncate update the metadata.
        # self.reader = BcolzMinuteBarReader(self.dest)

        assert self.writer.last_date_in_output_for_sid(sid) == days[0]

        _, last_close = self.trading_calendar.open_and_close_for_session(days[0])
        assert self.reader.last_available_dt == last_close

        minute = minutes[0]

        open_price = self.reader.get_value(sid, minute, "open")

        assert 10.0 == open_price

        high_price = self.reader.get_value(sid, minute, "high")

        assert 20.0 == high_price

        low_price = self.reader.get_value(sid, minute, "low")

        assert 30.0 == low_price

        close_price = self.reader.get_value(sid, minute, "close")

        assert 40.0 == close_price

        volume_price = self.reader.get_value(sid, minute, "volume")

        assert 50.0 == volume_price

    def test_truncate_all_data_points(self):

        tds = self.market_opens.index
        days = tds[
            tds.slice_indexer(
                start=self.test_calendar_start + timedelta(days=1),
                end=self.test_calendar_start + timedelta(days=3),
            )
        ]
        minutes = pd.DatetimeIndex(
            [
                self.market_opens[days[0]] + timedelta(minutes=60),
                self.market_opens[days[1]] + timedelta(minutes=120),
            ]
        )
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0, 11.0],
                "high": [20.0, 21.0],
                "low": [30.0, 31.0],
                "close": [40.0, 41.0],
                "volume": [50.0, 51.0],
            },
            index=minutes,
        )
        self.writer.write_from_sid_df_pairs("US", ((sid, data),))

        # Truncate to first day in the calendar, a day before the first
        # day with minute data.
        # self.writer.truncate(self.test_calendar_start)

        # Refresh the reader since truncate update the metadata.
        # self.reader = BcolzMinuteBarReader(self.dest)

        assert self.writer.last_date_in_output_for_sid(sid) == self.test_calendar_start

        cal = self.trading_calendar
        _, last_close = cal.open_and_close_for_session(self.test_calendar_start)
        assert self.reader.last_available_dt == last_close

    def test_early_market_close(self):
        """Test if writing non valid dates is allowed"""
        # Date to test is 2015-11-30 9:31
        # Early close is 2015-11-27 18:00
        friday_after_tday = pd.Timestamp("2015-11-27", tz="UTC")
        friday_after_tday_close = self.market_closes[friday_after_tday]

        before_early_close = friday_after_tday_close - timedelta(minutes=8)
        after_early_close = friday_after_tday_close + timedelta(minutes=8)

        monday_after_tday = pd.Timestamp("2015-11-30", tz="UTC")
        minute = self.market_opens[monday_after_tday]

        # Test condition where there is data written after the market
        # This will raise a not valid date error

        minutes = [before_early_close, after_early_close, minute]
        sid = 1
        data = pd.DataFrame(
            data={
                "open": [10.0, 11.0, np.nan],
                "high": [20.0, 21.0, np.nan],
                "low": [30.0, 31.0, np.nan],
                "close": [40.0, 41.0, np.nan],
                "volume": [50, 51, 0],
            },
            index=minutes,
        )
        with pytest.raises(NotValidDate):
            self.writer.write_from_sid_df_pairs(
                "US", ((sid, data),), exchange_name=self.trading_calendar.name
            )
