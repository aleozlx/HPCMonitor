.PHONY: run clean

SHELL=/bin/bash
PYTHON=python3
PYUIC=pyuic5

run: HPCMonitor.py mainwindow.py
	$(PYTHON) $<

clean:
	rm -f mainwindow.py

mainwindow.py: mainwindow.ui
	$(PYUIC) -o $@ $^

