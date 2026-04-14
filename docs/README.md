# TAD Documentation

This directory contains the Sphinx documentation for TAD (They All Die).

## Building the Documentation

### Prerequisites

Make sure you have the documentation dependencies installed:

```bash
conda activate tad-env  # If using conda
# or create a new environment
pip install sphinx furo myst-parser sphinx-codeautolink
```

### Building HTML Documentation

From this directory (`docs/`), run:

```bash
make html
```

Or on Windows:

```bash
make.bat html
```

The built HTML files will be in `_build/html/`. Open `_build/html/index.html` in a browser to view the documentation.

### Building Other Formats

Other output formats are supported:

```bash
make latex      # LaTeX files
make pdf        # PDF (requires pdflatex)
make epub       # EPUB ebook
make man        # Man pages
```

### Cleaning Build Files

To remove build artifacts:

```bash
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.md` - Main landing page
- `installation.md` - Installation instructions
- `tutorials.md` - Tutorial and example gallery
- `api/` - API reference documentation
- `examples/` - Example Python scripts (embedded in tutorials)

## Adding New Documentation

### Adding a Tutorial

1. Create a new Python script in `examples/` (e.g., `ex_11_myfeature.py`)
2. Include docstring and explanatory comments
3. Add a section to `tutorials.md` that includes the example using:
   ````
   ```{include} examples/ex_11_myfeature.py
   :code: python
   ```
   ````

### Adding API Documentation

API docs are automatically generated from NumPy-style docstrings in the source code via Sphinx autodoc. To document a new module:

1. Add a new section to `api/index.rst` with:
   ```rst
   .. automodule:: tad.mymodule
      :members:
      :undoc-members:
   ```

### Sphinx and MyST

- Documentation files can be written in **Markdown** (`.md`) or reStructuredText (`.rst`)
- Markdown files are processed by `myst-parser`
- Both formats can be mixed in a project

## Troubleshooting

### "sphinx-build: command not found"

Make sure Sphinx is installed and your environment is activated:

```bash
pip install sphinx
```

### Autodoc not finding modules

Ensure the package path is in `conf.py`:

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
```

### MyST extensions not working

Add extensions to `conf.py`:

```python
extensions = [
    "myst_parser",
    ...
]
myst_enable_extensions = ["colon_fence", "deflist"]
```

## Contributing to Docs

When contributing:

1. Use clear, concise language
2. Include working code examples
3. Add docstrings to all new functions (NumPy style)
4. Test your documentation builds locally with `make html`
5. Check that links and cross-references work

## References

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [Furo Theme](https://pradyunsg.me/furo/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)
