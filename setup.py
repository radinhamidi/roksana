from setuptools import setup, find_packages

setup(
    name='ROKSANA',
    version='0.2.1',
    author='Radin Hamidi Rad',
    author_email='radin.h@gmail.com',
    description='A toolkit for keyword search and attack methods on user-provided datasets.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme = 'README.md',
    url='https://github.com/radinhamidi/roksana',
    packages=find_packages(),
    include_package_data=True,
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
