import numpy as np
import pandas as pd
from zipline.data.hdf5_daily_bars import (
    HDF5BarReader,
    HDF5BarWriter,
    MultiCountryDailyBarReader,
)
from zipline.testing.predicates import assert_equal


class TestHDF5Writer:
    def test_write_empty_country(self, tmp_path):
        """
        Test that we can write an empty country to an HDF5 daily bar writer.

        This is useful functionality for some tests, but it requires a bunch of
        special cased logic in the writer.
        """
        path = tmp_path / "empty.h5"
        writer = HDF5BarWriter(path, date_chunk_size=30)
        writer.write_from_sid_df_pairs("US", iter(()), exchange_name="XNYS")

        reader = HDF5BarReader.from_path(path, "US")

        assert_equal(reader.sids, np.array([], dtype="int64"))

        empty_dates = np.array([], dtype="datetime64[ns]")
        assert_equal(reader.asset_start_dates, empty_dates)
        assert_equal(reader.asset_end_dates, empty_dates)
        assert_equal(reader.dates, empty_dates)

    def test_multi_country_attributes(self, tmp_path):
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

        writer.write("US", ohlcv(US), exchange_name="XNYS")
        writer.write("CA", ohlcv(CA), exchange_name="XTSE")

        reader = MultiCountryDailyBarReader.from_path(path)
        assert_equal(reader.countries, {"US", "CA"})
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
