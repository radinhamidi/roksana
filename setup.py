from setuptools import setup, find_packages
import os
import re

# Read the version from roksana/__init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "src", "roksana", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r'__version__ = ["\'](.+)["\']', line)
            if match:
                return match.group(1)
    raise RuntimeError("Version not found in roksana/__init__.py")


setup(
    name='ROKSANA',
    version=get_version(),
    author='Radin Hamidi Rad',
    author_email='radin.h@gmail.com',
    description='A toolkit for keyword search and attack methods on user-provided datasets.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/radinhamidi/roksana',
    package_dir={'':"src"},
    packages=find_packages("src"),
    # include_package_data=True,
    install_requires=[
        'torch',
        'torch-geometric',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
