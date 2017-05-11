.PHONY: run clean

SHELL=/bin/bash
PYTHON=python3
PYUIC=pyuic5

run: HPCMonitor.py mainwindow.py
	$(PYTHON) $<

clean:
	rm mainwindow.py

mainwindow.py: mainwindow.ui
	$(PYUIC) -o $@ $^

