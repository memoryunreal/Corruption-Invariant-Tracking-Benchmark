from setuptools import setup, find_packages

setup(
    name="mytest",
    version="1.0",
    author="silence",
    author_email="liz8@mail.sustech.edu.cn",
    description="Learn to Pack Python Module ",

    url="https://github.com/memoryunreal", 

    packages=find_packages(),

    install_requires = [
        'albumentations == 1.1.0',
        'imagecorruptions == 1.1.2',
        'opencv-python ==4.5.5.62'
    ]

)
