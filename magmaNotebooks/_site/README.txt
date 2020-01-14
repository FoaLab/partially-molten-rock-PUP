------------
REQUIREMENTS:
------------

- python >= 3.7 (inc. numpy, scipy, matplotlib)

- jupyter notebook >= 5.0.1

- activate the following jupyter notebook nbextensions (*):
	(some) LaTeX environments for Jupiter
	Hinterland
	Nbextensions edit menu item
	contrib_nbextensions_help_item
	Load TeX macros
	Nbextensions dashboard tab

(*) http://tljh.jupyter.org/en/latest/howto/admin/enable-extensions.html


-----------------------------------
HOW TO OPEN AND BUILD JUPYTER BOOKS
-----------------------------------

In the figs folder:

$ jupyter-book build magmaNotebooks --overwrite

In the magmaNotebooks folder, check if port is available

$ lsof -wni tcp:4000

Then build the website

$ bundle exec jekyll serve

Open the "Server address: http://127.0.0.1:4000/magmaNotes/" in the browser.
