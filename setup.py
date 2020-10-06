import setuptools

long_description = """Gemicai is at its core a deep learning library that makes working with PyToch. 
Gemicai also contains extensive functionality for the Dicom standard. 
This makes Gemicai an excellent tool for dealing with deep-learning-based medical image classifiers."""

setuptools.setup(
    name="gemicai",
    version="0.0.1",
    author="Gemicai",
    author_email="info@gemic.ai",
    description="Deep learning library for medical research ini PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gemicai/utilities",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
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
