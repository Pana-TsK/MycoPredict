from setuptools import setup, find_packages

setup(
    name='mycopredict',
    version='0.1',
    author='Panagiotis Tsampanis',
    author_email='panagiotis.tsampanis@ulb.be',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'mycopred = mycopred.main:run_pipeline',  # This points to the main function in mycopred.py
        ],
    },
    # Removed 'install_requires' to leave dependency management up to the user
)
