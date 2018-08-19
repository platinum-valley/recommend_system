
import unittest
from .. import recommend


class RecommendTest(unittest.TestCase):

    def test_all(self):
        """
        テストケース
        """
        model = recommend.Recommend("../dataset_norm.pickle")
        model.train()


if __name__ == "__main__":
    unittest.main()
