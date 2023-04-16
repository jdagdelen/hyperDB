from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hyperdb-python",
    version="0.1.1",
    author="John Dagdelen",
    author_email="jjdagdelen@gmail.com",
    description="A hyper-fast local vector database for use with LLM Agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdagdelen/hyperdb",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "openai"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.6',
)
