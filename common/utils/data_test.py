import unittest
from common.utils.data import *


class TestDataUtils(unittest.TestCase):
    def test_shard_empty_range(self):
        range = []
        shard_ranges = create_shard_ranges(range, shard_size=2)
        self.assertListEqual(shard_ranges, [])

        shard_ranges = create_shard_ranges(range, shard_size=0)
        self.assertListEqual(shard_ranges, [])

    def test_shard_one(self):
        range = [5]
        shard_ranges = create_shard_ranges(range, shard_size=2)
        self.assertListEqual(shard_ranges, [[5]])

        shard_ranges = create_shard_ranges(range, shard_size=256)
        self.assertListEqual(shard_ranges, [[5]])

        shard_ranges = create_shard_ranges(range, shard_size=1)
        self.assertListEqual(shard_ranges, [[5]])

    def test_shard_even_range(self):
        test_range = [3, 4, 5, 6, 7, 8]

        shard_ranges = create_shard_ranges(test_range, shard_size=6)
        self.assertListEqual(shard_ranges, [test_range])

        shard_ranges = create_shard_ranges(test_range, shard_size=3)
        self.assertListEqual(shard_ranges, [[3, 4, 5], [6, 7, 8]])

        shard_ranges = create_shard_ranges(test_range, shard_size=2)
        self.assertListEqual(shard_ranges, [[3, 4], [5, 6], [7, 8]])

        shard_ranges = create_shard_ranges(test_range, shard_size=1)
        self.assertListEqual(shard_ranges, [[3], [4], [5], [6], [7], [8]])

        shard_ranges = create_shard_ranges(test_range, shard_size=0)
        self.assertListEqual(shard_ranges, [])

    def test_shard_odd_range(self):
        test_range = [2, 3, 4, 5, 6, 7, 8]

        shard_ranges = create_shard_ranges(test_range, shard_size=6)
        self.assertListEqual(shard_ranges, [[2, 3, 4, 5, 6, 7], [8]])

        shard_ranges = create_shard_ranges(test_range, shard_size=4)
        self.assertListEqual(shard_ranges, [[2, 3, 4, 5], [6, 7, 8]])

        shard_ranges = create_shard_ranges(test_range, shard_size=3)
        self.assertListEqual(shard_ranges, [[2, 3, 4], [5, 6, 7], [8]])

        shard_ranges = create_shard_ranges(test_range, shard_size=2)
        self.assertListEqual(shard_ranges, [[2, 3], [4, 5], [6, 7], [8]])

    def test_get_group_and_idx(self):
        path = 'ha/foo_100.jpg'
        group, idx = get_group_and_idx(path)
        self.assertEqual('foo', group)
        self.assertEqual(100, idx)

    def test_get_group_and_idx_fail(self):
        path = 'ha/foo.jpg'
        group, idx = get_group_and_idx(path)
        self.assertEqual(None, group)
        self.assertEqual(None, idx)

if __name__ == '__main__':
    unittest.main()
