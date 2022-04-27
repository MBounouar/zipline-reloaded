import operator as op
import os
import re

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
import zipline.utils.paths as pth
from toolz import valmap
from zipline.assets import ASSET_DB_VERSION
from zipline.assets.asset_writer import check_version_info
from zipline.assets.synthetic import make_simple_equity_info
from zipline.data.bundles import (
    UnknownBundle,
    from_bundle_ingest_dirname,
    ingestions_for_bundle,
)
from zipline.data.bundles.core import (
    BadClean,
    _make_bundle_core,
    asset_db_path,
    to_bundle_ingest_dirname,
)
from zipline.lib.adjustment import Float64Multiply
from zipline.pipeline.loaders.synthetic import expected_bar_values_2d, make_bar_data
from zipline.testing import str_to_seconds
from zipline.utils.cache import dataframe_cache
from zipline.utils.calendar_utils import TradingCalendar, get_calendar

_1_ns = pd.Timedelta(1, unit="ns")


@pytest.fixture(scope="function")
def set_tmp_environ(request, tmp_path):
    tmpdir_path = str(tmp_path)
    request.cls.environ = {"ZIPLINE_ROOT": tmpdir_path}


@pytest.fixture
def make_bundle():
    (
        bundles,
        register,
        unregister,
        ingest,
        load,
        clean,
    ) = _make_bundle_core()
    yield bundles, register, unregister, ingest, load, clean

    for name in set(bundles.keys()):
        unregister(name)

    assert not bundles


@pytest.mark.usefixtures("set_tmp_environ")
class TestBundleCore:

    START_DATE = pd.Timestamp("2014-01-06", tz="utc")
    END_DATE = pd.Timestamp("2014-01-10", tz="utc")

    def test_register_decorator(self, make_bundle):
        (bundles, register, _, ingest, _, _) = make_bundle

        names = "abcde"
        for name in names:

            @register(name)
            def ingest(*args):
                pass

            assert name in bundles
            assert bundles[name].ingest is ingest
        assert set(bundles.keys()) == set(names)

    def test_register_call(self, make_bundle):
        (bundles, register, _, ingest, _, _) = make_bundle

        def ingest(*args):
            pass

        names = "abcde"
        for name in names:
            register(name, ingest)
            assert name in bundles
            assert bundles[name].ingest is ingest

        assert valmap(op.attrgetter("ingest"), bundles) == {k: ingest for k in names}
        assert set(bundles.keys()) == set(names)

    def test_register_no_create(self, make_bundle):
        (_, register, _, ingest, _, _) = make_bundle
        called = [False]

        @register("bundle", create_writers=False)
        def bundle_ingest(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            assert asset_db_writer is None
            assert minute_bar_writer is None
            assert daily_bar_writer is None
            assert adjustment_writer is None
            called[0] = True

        ingest("bundle", self.environ)
        assert called[0]

    def test_ingest(self, make_bundle):
        (_, register, _, ingest, load, _) = make_bundle
        calendar = get_calendar("XNYS")
        sessions = calendar.sessions_in_range(self.START_DATE, self.END_DATE)
        minutes = calendar.minutes_for_sessions_in_range(
            self.START_DATE,
            self.END_DATE,
        )

        sids = tuple(range(3))
        equities = make_simple_equity_info(
            sids,
            self.START_DATE,
            self.END_DATE,
        )

        daily_bar_data = make_bar_data(equities, sessions)
        minute_bar_data = make_bar_data(equities, minutes)
        first_split_ratio = 0.5
        second_split_ratio = 0.1
        splits = pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2014-01-08"),
                    "ratio": first_split_ratio,
                    "sid": 0,
                },
                {
                    "effective_date": str_to_seconds("2014-01-09"),
                    "ratio": second_split_ratio,
                    "sid": 1,
                },
            ]
        )

        @register(
            "bundle",
            calendar_name="NYSE",
            start_session=self.START_DATE,
            end_session=self.END_DATE,
        )
        def bundle_ingest(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            assert environ is self.environ

            asset_db_writer.write(equities=equities)

            minute_bar_writer.write_from_sid_df_pairs(
                "US", minute_bar_data, exchange_name=calendar.name
            )
            daily_bar_writer.write_from_sid_df_pairs(
                "US", daily_bar_data, exchange_name=calendar.name
            )

            adjustment_writer.write(splits=splits)

            assert isinstance(calendar, TradingCalendar)
            assert isinstance(cache, dataframe_cache)
            assert isinstance(show_progress, bool)

        ingest("bundle", environ=self.environ)
        bundle = load("bundle", environ=self.environ)

        assert set(bundle.asset_finder.sids) == set(sids)

        columns = "open", "high", "low", "close", "volume"

        actual = bundle.equity_minute_bar_reader.load_raw_arrays(
            columns,
            minutes[0],
            minutes[-1],
            sids,
        )

        for actual_column, colname in zip(actual, columns):
            np.testing.assert_array_equal(
                actual_column,
                expected_bar_values_2d(minutes, sids, equities, colname),
                err_msg=colname,
            )

        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            columns,
            self.START_DATE,
            self.END_DATE,
            sids,
        )
        for actual_column, colname in zip(actual, columns):
            np.testing.assert_array_equal(
                actual_column,
                expected_bar_values_2d(sessions, sids, equities, colname),
                err_msg=colname,
            )

        adjs_for_cols = bundle.adjustment_reader.load_pricing_adjustments(
            columns,
            sessions,
            pd.Index(sids),
        )
        for column, adjustments in zip(columns, adjs_for_cols[:-1]):
            # iterate over all the adjustments but `volume`
            assert adjustments == {
                2: [
                    Float64Multiply(
                        first_row=0,
                        last_row=2,
                        first_col=0,
                        last_col=0,
                        value=first_split_ratio,
                    )
                ],
                3: [
                    Float64Multiply(
                        first_row=0,
                        last_row=3,
                        first_col=1,
                        last_col=1,
                        value=second_split_ratio,
                    )
                ],
            }, column

        # check the volume, the value should be 1/ratio
        assert adjs_for_cols[-1] == {
            2: [
                Float64Multiply(
                    first_row=0,
                    last_row=2,
                    first_col=0,
                    last_col=0,
                    value=1 / first_split_ratio,
                )
            ],
            3: [
                Float64Multiply(
                    first_row=0,
                    last_row=3,
                    first_col=1,
                    last_col=1,
                    value=1 / second_split_ratio,
                )
            ],
        }, "volume"

    @pytest.mark.filterwarnings("ignore: Overwriting bundle with name")
    def test_ingest_assets_versions(self, make_bundle):
        (_, register, _, ingest, load, _) = make_bundle
        versions = (1, 2)

        called = [False]

        @register("bundle", create_writers=False)
        def bundle_ingest_no_create_writers(*args, **kwargs):
            called[0] = True

        now = pd.Timestamp.utcnow()
        with pytest.raises(ValueError, match="ingest .* creates writers .* downgrade"):
            ingest(
                "bundle",
                self.environ,
                assets_versions=versions,
                timestamp=now - pd.Timedelta(seconds=1),
            )
        assert not called[0]
        assert len(ingestions_for_bundle("bundle", self.environ)) == 1

        @register("bundle", create_writers=True)
        def bundle_ingest_create_writers(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            assert asset_db_writer is not None
            assert minute_bar_writer is not None
            assert daily_bar_writer is not None
            assert adjustment_writer is not None

            equities = make_simple_equity_info(
                tuple(range(3)),
                self.START_DATE,
                self.END_DATE,
            )
            asset_db_writer.write(equities=equities)
            called[0] = True

        # Explicitly use different timestamp; otherwise, test could run so fast
        # that first ingestion is re-used.
        ingest("bundle", self.environ, assets_versions=versions, timestamp=now)
        assert called[0]

        ingestions = ingestions_for_bundle("bundle", self.environ)
        assert len(ingestions) == 2

        for version in sorted(set(versions) | {ASSET_DB_VERSION}):
            eng = sa.create_engine(
                "sqlite:///"
                + asset_db_path(
                    "bundle",
                    to_bundle_ingest_dirname(ingestions[0]),  # most recent
                    self.environ,
                    version,
                )
            )
            metadata = sa.MetaData()
            metadata.reflect(eng)
            version_table = metadata.tables["version_info"]
            check_version_info(eng, version_table, version)

    @pytest.mark.parametrize("fnname", ["clean", "load"])
    def test_bundle_doesnt_exist(self, fnname, make_bundle):
        (_, _, _, _, load, clean) = make_bundle
        with pytest.raises(
            UnknownBundle, match="No bundle registered with the name 'ayy'"
        ):
            eval(fnname)("ayy", environ=self.environ)

    def test_load_no_data(self, make_bundle):
        (_, register, _, _, load, _) = make_bundle
        # register but do not ingest data
        register("bundle", lambda *args: None)

        ts = pd.Timestamp("2014", tz="UTC")
        expected_msg = "no data for bundle 'bundle' on or before %s" % ts
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            load("bundle", timestamp=ts, environ=self.environ)

    def _list_bundle(self):
        return {
            os.path.join(pth.data_path(["bundle", d], environ=self.environ))
            for d in os.listdir(
                pth.data_path(["bundle"], environ=self.environ),
            )
        }

    def _empty_ingest(self, bundles, register, ingest, _wrote_to=[]):
        """Run the nth empty ingest.

        Returns
        -------
        wrote_to : str
            The timestr of the bundle written.
        """
        if not bundles:

            @register(
                "bundle",
                calendar_name="NYSE",
                start_session=pd.Timestamp("2014", tz="UTC"),
                end_session=pd.Timestamp("2014", tz="UTC"),
            )
            def _(
                environ,
                asset_db_writer,
                minute_bar_writer,
                daily_bar_writer,
                adjustment_writer,
                calendar,
                start_session,
                end_session,
                cache,
                show_progress,
                output_dir,
            ):
                _wrote_to.append(output_dir)

        _wrote_to[:] = []
        ingest("bundle", environ=self.environ)
        assert len(_wrote_to) == 1, "ingest was called more than once"
        ingestions = self._list_bundle()
        assert _wrote_to[0] in ingestions, "output_dir was not in the bundle directory"

        return _wrote_to[0]

    def test_clean_keep_last(self, make_bundle):
        (bundles, register, _, ingest, _, clean) = make_bundle
        first = self._empty_ingest(bundles, register, ingest)

        assert clean("bundle", keep_last=1, environ=self.environ) == set()
        assert self._list_bundle() == {first}, "directory should not have changed"

        second = self._empty_ingest(bundles, register, ingest)
        assert self._list_bundle() == {first, second}, "two ingestions are not present"
        assert clean("bundle", keep_last=1, environ=self.environ) == {first}
        assert self._list_bundle() == {
            second
        }, "first ingestion was not removed with keep_last=2"

        third = self._empty_ingest(bundles, register, ingest)
        fourth = self._empty_ingest(bundles, register, ingest)
        fifth = self._empty_ingest(bundles, register, ingest)

        assert self._list_bundle() == {
            second,
            third,
            fourth,
            fifth,
        }, "larger set of ingestions did not happen correctly"

        assert clean("bundle", keep_last=2, environ=self.environ) == {
            second,
            third,
        }

        assert self._list_bundle() == {
            fourth,
            fifth,
        }, "keep_last=2 did not remove the correct number of ingestions"

        with pytest.raises(BadClean):
            clean("bundle", keep_last=-1, environ=self.environ)

        assert self._list_bundle() == {
            fourth,
            fifth,
        }, "keep_last=-1 removed some ingestions"

        assert clean("bundle", keep_last=0, environ=self.environ) == {
            fourth,
            fifth,
        }

        assert (
            self._list_bundle() == set()
        ), "keep_last=0 did not remove the correct number of ingestions"

    @staticmethod
    def _ts_of_run(run):
        return from_bundle_ingest_dirname(run.rsplit(os.path.sep, 1)[-1])

    def test_clean_before_after(self, make_bundle):
        (bundles, register, _, ingest, _, clean) = make_bundle
        first = self._empty_ingest(bundles, register, ingest)
        assert (
            clean(
                "bundle",
                before=self._ts_of_run(first),
                environ=self.environ,
            )
            == set()
        )
        assert self._list_bundle() == {
            first
        }, "directory should not have changed (before)"

        assert (
            clean(
                "bundle",
                after=self._ts_of_run(first),
                environ=self.environ,
            )
            == set()
        )

        assert self._list_bundle() == {
            first
        }, "directory should not have changed (after)"

        assert clean(
            "bundle",
            before=self._ts_of_run(first) + _1_ns,
            environ=self.environ,
        ) == {first}
        assert self._list_bundle() == set(), "directory now be empty (before)"

        second = self._empty_ingest(bundles, register, ingest)
        assert clean(
            "bundle",
            after=self._ts_of_run(second) - _1_ns,
            environ=self.environ,
        ) == {second}

        assert self._list_bundle() == set(), "directory now be empty (after)"

        third = self._empty_ingest(bundles, register, ingest)
        fourth = self._empty_ingest(bundles, register, ingest)
        fifth = self._empty_ingest(bundles, register, ingest)
        sixth = self._empty_ingest(bundles, register, ingest)

        assert self._list_bundle() == {
            third,
            fourth,
            fifth,
            sixth,
        }, "larger set of ingestions did no happen correctly"

        assert clean(
            "bundle",
            before=self._ts_of_run(fourth),
            after=self._ts_of_run(fifth),
            environ=self.environ,
        ) == {third, sixth}

        assert self._list_bundle() == {
            fourth,
            fifth,
        }, "did not strip first and last directories"
