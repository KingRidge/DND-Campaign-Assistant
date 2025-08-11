from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="dnd-campaign-assistant",
    # version is handled by setuptools_scm
    author="KingRidge",
    description="A D&D campaign assistant with AI-powered content generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KingRidge/dnd-campaign-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment :: Role-Playing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    package_data={
        "dnd_campaign_assistant": [
            "data/*.json",
            "data/*.md",
            "data/*.txt",
            "data/locations/*",
            "data/notes/*",
            "data/npcs/*",
            "data/scenes/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "dnd-assistant=dnd_campaign_assistant.campaign_assistant:main",
        ],
    },
)
