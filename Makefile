SOURCE ?= slides

PDF_READER ?= open
PDFLATEX ?= pdflatex -halt-on-error -shell-escape
BIBTEX ?= bibtex

all: clean report

report:
	$(PDFLATEX) $(SOURCE).tex
	$(BIBTEX) $(SOURCE)
	$(PDFLATEX) $(SOURCE).tex
	$(PDFLATEX) $(SOURCE).tex

partial:
	$(PDFLATEX) $(SOURCE).tex

view: report
	@$(PDF_READER) $(SOURCE).pdf &

log:
	@open $(SOURCE).log

.PHONY: clean

clean:
	rm -f *.log *.out *.aux *.bbl *.blg $(SOURCE).pdf
