# -*- coding: utf-8 -*-
"""Plugin to allow viewcode to work for inherited methods.

Created on March 13, 2025
@author: Donald Erb

"""

from sphinx.pycode import ModuleAnalyzer


def find_super_method(app, modname):
    """
    Hooks into the sphinx.ext.viewcode extension to allow viewing code for inherited methods.

    Documents the locations of the methods of the super classes of Baseline and Baseline2D
    and tells viewcode that these locations also belong to the methods of Baseline or Baseline2D.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application object.
    modname : str or None
        The name of the module to create tags for, such as "pybaselines.spline" and
        "pybaselines.two_d.spline".

    Returns
    -------
    code : str
        The entire source code for the specified module.
    output_tags : dict[str, tuple(str, int, int)]
        A dictionary that maps the name of each object in the module to its type ('def'
        for functions and methods, 'class' for classes, and 'other' for attributes) and the
        start and end line numbers in ``code``. For example, the key "Baseline.poly" would
        have a value resembling `('def', 90, 140))`, and the key "Baseline" would have a value
        resembling `('class', 14, 80)`.

    Notes
    -----
    The "viewcode_follow_imported_members" toggle for sphinx.ext.viewcode can find the
    locations of inherited members, but it does not link to the corresponding inherited methods
    since when it finds the methods of the super class, it does not add the subclass as an
    appropriate key for the file. For example, if it were looking for "pybaselines.Baseline.asls",
    it would correctly identify the file location as "pybaselines.whittaker", but since the only
    keys within that file correspond to "_Whittaker" (eg. "_Whittaker.asls" in this example),
    viewcode does not think that "Baseline.asls" exists in the file and does not link to it.

    In order to fix the above problem, have to hook into the earlier event
    "viewcode-find-source" that defines the objects for each file and add the subclass
    to the output tags so that when viewcode then follows the imports, it correctly finds
    the inherited methods and associates them with the subclass.

    For an example of what ``ModuleAnalyzer`` tags and what viewcode expects, see the following
    issue from Sphinx: https://github.com/sphinx-doc/sphinx/issues/11279.

    """
    if modname is None:
        return

    modules = (
        'classification', 'misc', 'morphological', 'optimizers', 'polynomial', 'spline',
        'smooth', 'whittaker'
    )
    for module in modules:
        if module in modname:
            class_obj = f'_{module.capitalize()}'
            break
    else:
        # let viewcode handle other modules that don't need to add Baseline or Baseline2D objects
        return

    try:
        analyzer = ModuleAnalyzer.for_module(modname)
        analyzer.find_tags()
        code = analyzer.code
        tags = analyzer.tags
    except Exception:
        return

    if 'two_d' in modname:
        new_obj = 'Baseline2D'
    else:
        new_obj = 'Baseline'

    output_tags = {}
    for key, val in tags.items():
        new_key = key.replace(class_obj, new_obj)
        if new_key != new_obj:
            # don't want to define Baseline or Baseline2D outside of the api module
            output_tags[new_key] = val
        output_tags[key] = val  # keep a reference to the original class as well
    return (code, output_tags)


def setup(app):
    """Connects the find_super_method to the viewcode-find-source event.

    Returns
    -------
    dict
        Relevant information about the extension to pass to Sphinx. See
        https://www.sphinx-doc.org/en/master/extdev/index.html for metadata fields.

    """
    app.connect('viewcode-find-source', find_super_method)
    # according to https://www.sphinx-doc.org/en/master/extdev/index.html, since this
    # extension does not store data in the Sphinx environment, and it is not doing anything
    # that is not parallel safe, then parallel_read_safe can be set to True; plus
    # sphinx.ext.viewcode is set as True for parallel_read_safe, so the same logic should apply
    return {
        'version': '0.0.1',
        'parallel_read_safe': True,
    }
