import re
import pprint

mathjax_config = { 'TeX': {'Macros': {}}}

with open('latexdefs.tex', 'r') as f:
    for line in f:
        macros = re.findall(r'\\(DeclareRobustCommand|newcommand){\\(.*?)}(\[(\d)\])?{(.+)}', line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax_config['TeX']['Macros'][macro[1]] = "{"+macro[4]+"}"
            else:
                mathjax_config['TeX']['Macros'][macro[1]] = ["{"+macro[4]+"}", int(macro[3])]

pprint.pprint(mathjax_config)