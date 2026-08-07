import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "PyPulseq-Star"
author = "Sairam Geethanath"
release = "0.2.0a1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "folder_structure_critique.md",
]

html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}