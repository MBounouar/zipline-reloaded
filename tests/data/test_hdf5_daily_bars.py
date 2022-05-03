from re import A
import numpy as np
import pandas as pd
from zipline.data.hdf5_daily_bars import (
    HDF5BarReader,
    HDF5BarWriter,
    MultiExchangeDailyBarReader,
)
from zipline.testing.predicates import assert_equal
from zipline.utils.calendar_utils import get_calendar


class TestHDF5Writer:
    def test_write_empty_exchange(self, tmp_path):
        """
        Test that we can write an empty exchange to an HDF5 daily bar writer.

        This is useful functionality for some tests, but it requires a bunch of
        special cased logic in the writer.
        """
        path = tmp_path / "empty.h5"
        writer = HDF5BarWriter(path, date_chunk_size=30)
        writer.write_from_sid_df_pairs("XNYS", iter(()))

        reader = HDF5BarReader.from_path(path, "XNYS")
        calendar = reader.trading_calendar
        assert calendar == get_calendar("XNYS")
        assert_equal(reader.sids, np.array([], dtype="int64"))

        empty_dates = np.array([], dtype="datetime64[ns]")
        assert_equal(reader.asset_start_dates, empty_dates)
        assert_equal(reader.asset_end_dates, empty_dates)
        assert_equal(reader.dates, empty_dates)

    def test_multi_exchange_attributes(self, tmp_path):
        path = tmp_path / "multi.h5"
        writer = HDF5BarWriter(path, date_chunk_size=30)

        US = pd.DataFrame(
            data=np.ones((3, 5)),
            index=pd.to_datetime(["2014-01-02", "2014-01-03", "2014-01-06"]),
            columns=np.arange(1, 6),
        )

        CA = pd.DataFrame(
            data=np.ones((2, 4)) * 2,
            index=pd.to_datetime(["2014-01-03", "2014-01-07"]),
            columns=np.arange(100, 104),
        )

        def ohlcv(frame):
            return {
                "open": frame,
                "high": frame,
                "low": frame,
                "close": frame,
                "volume": frame,
            }

        writer.write(
            exchange_code="XNYS",
            frames=ohlcv(US),
        )
        writer.write(exchange_code="XTSE", frames=ohlcv(CA))

        reader = MultiExchangeDailyBarReader.from_path(path)

        assert_equal(reader.exchanges, {"XNYS", "XTSE"})
        assert reader.trading_calendar == [get_calendar("XNYS"), get_calendar("XTSE")]
        assert_equal(
            reader.sessions,
            pd.to_datetime(
                [
                    "2014-01-02",
                    "2014-01-03",
                    "2014-01-06",
                    "2014-01-07",
                ],
                utc=True,
            ),
        )
