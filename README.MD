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

In the `partially-molten-rock-PUP` folder, the command below will build the HTML files based on the `contents` folder:
```
$ jupyter-book build --overwrite ../partially-molten-rock-PUP
```

In order to publish the changes, just type
```
$ ghp-import -p -f _build 
```