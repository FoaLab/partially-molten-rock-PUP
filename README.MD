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

In the `partially-molten-rock-PUP` folder, type

```
$ jupyter-book build .
```
and follow the command-line intructions.

Next, publish the book online with GitHub pages

```
ghp-import -n -p -f partially-molten-rock-PUP/_build/html
```

Developed by Frederico Teixeira.