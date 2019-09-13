import io
import clize
import clize.help
import sigtools.wrappers

# dirty hack to fix terminal width in a notebook
# (not needed in an actual terminal)
clize.util.get_terminal_width = lambda: 78

# help helpers

def docutilize(obj):
    from  inspect import cleandoc
    from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring
    if type(obj) == str:
        doc = obj
    else:
        doc = obj.__doc__
    doc = cleandoc(doc)
    doc = str(NumpyDocstring(doc))
    doc = str(GoogleDocstring(doc))
    # exception and keyword markup seems to trip up the docutils parser
    doc = doc.replace(':exc:', '')
    doc = doc.replace(':keyword', ':param')
    doc = doc.replace(':kwtype', ':type')
    if type(obj) == str:
        return doc
    obj.__doc__ = doc
    return obj

class HelpForNapoleonDocstring(clize.help.HelpForAutodetectedDocstring):
    def add_docstring(self, docstring, *args, **kwargs):
        docstring = docutilize(docstring)
        super().add_docstring(docstring, *args, **kwargs)

class DocutilizeClizeHelp(clize.help.ClizeHelp):
    def __init__(self, subject, owner,
                 builder=HelpForNapoleonDocstring.from_subject):
        super().__init__(subject, owner, builder)

# converters

def maybe_coerce_with(conv, obj):
    """Apply converter if str, pass through otherwise."""
    return conv(obj) if isinstance(obj, str) else obj

@clize.parser.value_converter
def inputcube(input):
    from improver.utilities.load import load_cube
    return maybe_coerce_with(load_cube, input)

@clize.parser.value_converter
def inputjson(input):
    from improver.utilities.cli_utilities import load_json_or_none
    return maybe_coerce_with(load_json_or_none, input)

# output handling

def outputcube(cube, output):
    """Save Cube. Used as annotation of return values."""
    from improver.utilities.save import save_netcdf
    if output:
        save_netcdf(cube, output)
    return cube

def save_at_index(index, outfile, func, *args, **kwargs):
    """Helper function to save one value out of multiple returned."""
    result = result_selection = func(*args, **kwargs)
    saver = func.__annotations__.get('return')
    if not outfile or not saver:
        return result
    if type(saver) == tuple:
        saver = saver[index]
        result_selection = result[index]
    elif index:
        raise IndexError('Non-zero index is valid only for multiple return values.')
    if not isinstance(index, slice):
        result_selection, outfile = (result_selection,), (outfile,)
    if len(result_selection) != len(outfile):
        raise ValueError('Number of selected results does not match return annotation.')
    for res, out in zip(result_selection, outfile):
        saver(res, out)
    return result

@sigtools.wrappers.decorator
def with_output(wrapped, *args, output=None, **kwargs):
    """
    :param output: Output file name
    """
    return save_at_index(0, output, wrapped, *args, **kwargs)

# cli object creation and handling

def clizefy(func=None, with_output=with_output,
            helper_class=DocutilizeClizeHelp, **kwargs):
    """Decorator for creating CLI objects."""
    from functools import partial
    if func is None:
        return partial(clizefy, with_output=with_output,
                       helper_class=helper_class, **kwargs)
    if with_output:
        func = with_output(func)
    func = clize.Clize.keep(func, helper_class=helper_class, **kwargs)
    return func
