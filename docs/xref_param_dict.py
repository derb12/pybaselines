# -*- coding: utf-8 -*-
"""Plugin to add cross references for types within returned parameter dictionaries.

Created on January 25, 2026
@author: Donald Erb

"""

from numpydoc.xref import make_xref


def xref_parameter_dict(app, what, name, obj, options, lines):
    """
    Adds cross references for items within the parameter dictionaries for baseline methods.

    Makes it so that the types for items within the parameter dictionary are also cross
    referenced just like the inputs and returns are. See
    https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#event-autodoc-process-docstring
    for more details on the Sphinx event. This should be ran after numpydoc has ran since it
    expects numpydoc formatting within the docstrings.

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

    Raises
    ------
    ValueError
        Raised if the formatting for a parameter item is incorrect.

    Notes
    -----
    Relies on numpydoc formatting within the sections, so this may break for future numpydoc
    releases. For reference, tested on numpydoc versions 1.8.0, 1.9.0, and 1.10.0.

    """
    # parameter dictionary items are only within Baseline[2D] methods or the functional
    # interface
    if what not in ('method', 'function'):
        return

    # if using napoleon instead of numpydoc, this should raise an error, which is desired behavior
    # since docstring formatting will not be the same otherwise
    xref_aliases = app.config.numpydoc_xref_aliases_complete
    xref_ignore = app.config.numpydoc_xref_ignore

    key_type_separator = ': '
    in_return_section = False
    for i, line in enumerate(lines):
        # NOTE this could be made more robust using regular expressions, but the formatting
        # should be kept simple since this is just for internal formatting
        #
        # Assumes formatting within the Returns sections is like:
        #
        # params : dict
        #     * 'key' : value_typing
        #         Description of value.
        if in_return_section and line.startswith('        *'):
            key, *value_typing = line.split(key_type_separator)
            if not value_typing:
                # in case something is incorrectly formatted as "name type" rather than
                # "name : type"
                raise ValueError(
                    f'Incorrect parameter dictionary format for {name} at line: "{line}"'
                )
            # could split value_typing[0] using ',' to separate things like
            # "numpy.ndarray, shape (N,)", but that fails for others like "dict[str, list]",
            # so just pass the full typing reference to numpydoc and let it process accordingly
            xref = make_xref(value_typing[0], xref_aliases=xref_aliases, xref_ignore=xref_ignore)
            lines[i] = key_type_separator.join([key, xref])
        elif in_return_section and (line.startswith(':') or line.startswith('..')):
            # other docstring sections after Returns start with things like '.. rubric:: References'
            # or ':Raises:' after numpydoc formatting
            break
        elif line == ':Returns:':
            in_return_section = True


def setup(app):
    """Connects the xref_parameter_dict to the autodoc-process-docstring event.

    Returns
    -------
    dict
        Relevant information about the extension to pass to Sphinx. See
        https://www.sphinx-doc.org/en/master/extdev/index.html for metadata fields.

    """
    # Add a high priority so that numpydoc should process the docstrings first
    app.connect('autodoc-process-docstring', xref_parameter_dict, priority=10000)
    # according to https://www.sphinx-doc.org/en/master/extdev/index.html, since this
    # extension does not store data in the Sphinx environment, and it is not doing anything
    # that is not parallel safe, then parallel_read_safe can be set to True; plus
    # numpydoc is set as True for parallel_read_safe, so the same logic should apply
    return {
        'version': '0.0.1',
        'parallel_read_safe': True,
    }
