import setuptools

long_description = '''Gemicai is an open-source deep-learning library with extensive functionality for the Dicom 
standard. Its functionalities and applicability make Gemicai an excellent tool for research concerning artificial 
intelligence in medical imaging. A key benefit of Gemicai is making data pre-processing practical and easy to use while 
automatically configuring the GPU hardware acceleration. This enables researchers to save time on ordinary tasks so that 
they can focus on more important matters. '''

setuptools.setup(
    name='Gemicai',
    version='0.5.0',
    author='Gemicai',
    author_email='info@gemic.ai',
    description='Deep learning library for medical imaging',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gemicai/utilities',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pydicom>=2.0.0,<2.1',
        'torch>=1.6.0,<1.7',
        'torchvision>=0.7.0,<0.8',
        'matplotlib',
        'pandas',
    ]
)
