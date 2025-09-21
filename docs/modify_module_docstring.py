# -*- coding: utf-8 -*-
"""Plugin to remove all but the first lines of a module docstring for documentation.

Created on August 5, 2025
@author: Donald Erb

"""


def condense_module_docstring(app, what, name, obj, options, lines):
    """
    Removes all but the first line from module docstrings.

    The documentation looks nicer for modules when other header information is removed such as
    third-party licenses. See
    https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring
    for more details on the Sphinx event.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application object.
    what : str
        The type of object; can be 'module', 'class', etc.
    name : str
        The full name of the object being documented.
    obj : object
        The actual object, eg. the class, function, module, etc. being documented.
    options : sphinx.ext.autodoc.Options
        The options for the current object's directive.
    lines : list[str]
        The list of strings for the docstring. Must be modified in-place in order
        to change the value.

    """
    if what == 'module':
        del lines[1:]


def setup(app):
    """Connects the condense_module_docstring to the autodoc-process-docstring event.

    Returns
    -------
    dict
        Relevant information about the extension to pass to Sphinx. See
        https://www.sphinx-doc.org/en/master/extdev/index.html for metadata fields.

    """
    app.connect('autodoc-process-docstring', condense_module_docstring)
    # since this modifies the docstrings in-place, safer to say that it is not parallel-safe
    return {
        'version': '0.0.1',
        'parallel_read_safe': False,
    }
