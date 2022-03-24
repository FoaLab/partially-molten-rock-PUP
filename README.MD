# The Dynamics of Partially Molten Rocks

By Richard F. Katz

University of Oxford

Published by Princeton University Press

https://press.princeton.edu/books/hardcover/9780691176567/the-dynamics-of-partially-molten-rock

This online supplement can be cited as follows: Teixeira & Katz (2021), _Online Supplement to The Dynamics of Partially Molten Rock_.  DOI: XXXX.

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
$ jupyter-book build --path-output . content
```
and follow the command-line intructions.

Next, publish the book online with GitHub pages

```
ghp-import -n -p -f _build/html
```
