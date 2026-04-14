"""
Configuration file for the Sphinx documentation builder.
"""

import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Project information
project = "TAD (They All Die)"
copyright = "2026, Florian Kolbl, Jaderson Polli"
author = "Florian Kolbl, Jaderson Polli"
release = "1.0.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Support NumPy-style docstrings
    "sphinx.ext.viewcode",
    "myst_parser",  # Markdown support
]

# MyST configuration
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 2

# Autodoc settings
autodoc_typehints = "description"
autosummary_generate = True
autosummary_generate_overwrite = False

# Napoleon settings (for NumPy-style docstrings)
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Theme
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1f77b4",
        "color-brand-content": "#1f77b4",
    },
}

# Source file extensions
source_suffix = {
    ".rst": None,
    ".md": None,
}

# Templates path
templates_path = ["_templates"]

# HTML output
html_static_path = []

# Logo and title
html_title = "TAD Documentation"
html_logo = None

# Autodoc mock imports (if needed)
autodoc_mock_imports = []

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
