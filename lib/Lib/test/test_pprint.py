import pprint
import unittest

import test_support

try:
    uni = unicode
except NameError:
    def uni(x):return x


class QueryTestCase(unittest.TestCase):

    def setUp(self):
        self.a = range(100)
        self.b = range(200)
        self.a[-12] = self.b

    def test_basic(self):
        """Verify .isrecursive() and .isreadable() w/o recursion."""
        verify = self.assert_
        for safe in (2, 2.0, 2j, "abc", [3], (2,2), {3: 3}, uni("yaddayadda"),
                     self.a, self.b):
            verify(not pprint.isrecursive(safe),
                   "expected not isrecursive for " + `safe`)
            verify(pprint.isreadable(safe),
                   "expected isreadable for " + `safe`)

    def test_knotted(self):
        """Verify .isrecursive() and .isreadable() w/ recursion."""
        # Tie a knot.
        self.b[67] = self.a
        # Messy dict.
        self.d = {}
        self.d[0] = self.d[1] = self.d[2] = self.d

        verify = self.assert_

        for icky in self.a, self.b, self.d, (self.d, self.d):
            verify(pprint.isrecursive(icky), "expected isrecursive")
            verify(not pprint.isreadable(icky),  "expected not isreadable")

        # Break the cycles.
        self.d.clear()
        del self.a[:]
        del self.b[:]

        for safe in self.a, self.b, self.d, (self.d, self.d):
            verify(not pprint.isrecursive(safe),
                   "expected not isrecursive for " + `safe`)
            verify(pprint.isreadable(safe),
                   "expected isreadable for " + `safe`)

    def test_unreadable(self):
        """Not recursive but not readable anyway."""
        verify = self.assert_
        for unreadable in type(3), pprint, pprint.isrecursive:
            verify(not pprint.isrecursive(unreadable),
                   "expected not isrecursive for " + `unreadable`)
            verify(not pprint.isreadable(unreadable),
                   "expected not isreadable for " + `unreadable`)

    def test_same_as_repr(self):
        "Simple objects and small containers that should be same as repr()."
        verify = self.assert_
        for simple in (0, 0L, 0+0j, 0.0, "", uni(""), (), [], {}, verify, pprint,
                       -6, -6L, -6-6j, -1.5, "x", uni("x"), (3,), [3], {3: 6},
                       (1,2), [3,4], {5: 6, 7: 8},
                       {"xy\tab\n": (3,), 5: [[]], (): {}},
                       range(10, -11, -1)
                      ):
            native = repr(simple)
            for function in "pformat", "saferepr":
                f = getattr(pprint, function)
                got = f(simple)
                verify(native == got, "expected %s got %s from pprint.%s" %
                                      (native, got, function))


    def test_basic_line_wrap(self):
        """verify basic line-wrapping operation"""
        o = {'RPM_cal': 0,
             'RPM_cal2': 48059,
             'Speed_cal': 0,
             'controldesk_runtime_us': 0,
             'main_code_runtime_us': 0,
             'read_io_runtime_us': 0,
             'write_io_runtime_us': 43690}
        exp = """\
{'RPM_cal': 0,
 'RPM_cal2': 48059,
 'Speed_cal': 0,
 'controldesk_runtime_us': 0,
 'main_code_runtime_us': 0,
 'read_io_runtime_us': 0,
 'write_io_runtime_us': 43690}"""
        self.assertEqual(pprint.pformat(o), exp)

def test_main():
    test_support.run_unittest(QueryTestCase)


if __name__ == "__main__":
    test_main()
