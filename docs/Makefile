# Makefile for Sphinx documentation
#

# You can set these variables from the command line:
#   SPHINXOPTS    = -W --keep-going (warnings as errors; keep going)
#   SPHINXBUILD   = sphinx-build
#   PAPER         = a4
#   BUILDDIR      = _build

# User-friendly check for sphinx-build
ifeq ($(shell which sphinx-build >/dev/null 2>&1; echo $$?), 1)
$(error The 'sphinx-build' command was not found. Make sure you have Sphinx installed.)
endif

SPHINXBUILD   ?= sphinx-build
PAPER         ?= a4
BUILDDIR      ?= _build

# Put it first so that "make" without argument is like "make help"
.PHONY: help Makefile

help:
	@$(SPHINXBUILD) -M help "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" intro.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: html
html:
	@$(SPHINXBUILD) -M html "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Build finished. The HTML files are in $(BUILDDIR)/html"
