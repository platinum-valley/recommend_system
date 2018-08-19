
import unittest
from recommend_system import recommend


class RecommendTest(unittest.TestCase):

    def test_all(self):
        """
        テストケース
        """
        model = recommend.Recommend()
        model.train()


if __name__ == "__main__":
    unittest.main()
