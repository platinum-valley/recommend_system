from setuptools import setup


setup(
    name="recommend_system",
    version="0.0.1",
    description="recommend movie from movielens10k by NN method",
    author="shinichiro tamiya",
    author_email='s163351@edu.tut.ac.jp',
    url="https://github.com/platinum-valley",
    install_requires=["numpy", "scipy", "scikit-learn"],
    license=license,
    test_suite="tests"
)
