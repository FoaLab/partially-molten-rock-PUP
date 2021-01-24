# The Dynamics of Partially Molten Rocks

By Prof. Dr. Richard Katz
University of Oxford

## Requirements:

* python >= 3.7 (inc. numpy, scipy, matplotlib)
* jupyter notebook >= 5.0.1
* activate the following jupyter notebook nbextensions, see details [here](http://tljh.jupyter.org/en/latest/howto/admin/enable-extensions.html):
  * (some) LaTeX environments for Jupiter
  * Hinterland
  * Nbextensions edit menu item
  * contrib_nbextensions_help_item
  * Load TeX macros
  * Nbextensions dashboard tab

- [Jupyter Book](https://jupyterbook.org/intro.html) link

## How to open and build Jupyter books

In the figs folder:
```
$ jupyter-book build magmaNotebook --overwrite
```

In the magmaNotebook folder, check if port is available
```
$ lsof -wni tcp:4000
```
Then build the website
```
$ bundle exec jekyll serve
```
Open the server address: http://127.0.0.1:4000/magmaNotes/ in the browser.
