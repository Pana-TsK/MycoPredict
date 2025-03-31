from setuptools import setup, find_packages

setup(
    name='mycopredict',
    version='0.1',
    author='Panagiotis Tsampanis',
    author_email='panagiotis.tsampanis@ulb.be',
    packages=find_packages(),
    install_requires=[
        'chemprop',  # list any other dependencies here
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'mycopred = mycopred.main:run_pipeline',  # This points to the main function in mycopred.py
        ],
    },
)