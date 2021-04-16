from copy import copy, deepcopy
from pickle import loads, dumps
import sys
from unittest import TestCase
from weakref import ref

from zipline.utils.sentinel import sentinel


class SentinelTestCase(TestCase):
    def tearDown(self):
        sentinel._cache.clear()  # don't pollute cache.

    def test_name(self):
        assert sentinel("a").__name__ ==  "a"

    def test_doc(self):
        assert sentinel("a", "b").__doc__ == "b"

    def test_doc_differentiates(self):
        # the following assignment must be exactly one source line above
        # the assignment of ``a``.
        line = sys._getframe().f_lineno
        a = sentinel("sentinel-name", "original-doc")
        with self.assertRaises(ValueError) as e:
            sentinel(a.__name__, "new-doc")

        msg = str(e.exception)
        assert a.__name__ in msg
        assert a.__doc__ in  msg
        # strip the 'c' in case ``__file__`` is a .pyc and we are running this
        # test twice in the same process...
        assert "%s:%s" % (__file__.rstrip("c"), line + 1) in msg

    def test_memo(self):
        assert sentinel("a") is sentinel("a")

    def test_copy(self):
        a = sentinel("a")
        assert copy(a) is a

    def test_deepcopy(self):
        a = sentinel("a")
        assert deepcopy(a) is a

    def test_repr(self):
        assert repr(sentinel("a")) == "sentinel('a')"

    def test_new(self):
        with self.assertRaises(TypeError):
            type(sentinel("a"))()

    def test_pickle_roundtrip(self):
        a = sentinel("a")
        assert loads(dumps(a)) is a

    def test_weakreferencable(self):
        ref(sentinel("a"))
