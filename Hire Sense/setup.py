from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hiresense-ai",
    version="1.0.0",
    author="HireSense Contributors",
    description="Resume-to-Job Match Engine for Recruiters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
)
