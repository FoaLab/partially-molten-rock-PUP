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

- useful links
https://jupyterbook.org/guide/02_create.html
https://jupyterbook.org/guide/03_build.html#Create-an-online-repository-for-your-book
https://jupyterbook.org/guide/publish/github-pages.html

And perhaps

https://jupyterbook.org/guide/publish/book-html.html#install-ruby-plugins

-----------------------------------
HOW TO OPEN AND BUILD JUPYTER BOOKS
-----------------------------------

In the figs folder:

$ jupyter-book build magmaNotebook --overwrite

In the magmaNotebook folder, check if port is available

$ lsof -wni tcp:4000

Then build the website

$ bundle exec jekyll serve

Open the server address: "http://127.0.0.1:4000/magmaNotes/" in the browser.
