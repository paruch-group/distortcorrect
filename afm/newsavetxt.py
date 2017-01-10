# DO NOT CHANGE THIS FILE!
#
# This file contains the savetxthd function, which is slightly modified
# version of the savetxt function from the numpy library.  The only change
# is the option of putting a header at the beginning of the text file.
# To use savetxthd,  the first line of your program should
# be "from newsavetext import *".
#
# This modification was proposed at
#    http://projects.scipy.org/numpy/ticket/1079
# and may appear in future versions of the numpy library. 


import numpy as np
import itertools

# Adapted from matplotlib

def _getconv(dtype):
    typ = dtype.type
    if issubclass(typ, np.bool_):
        return lambda x: bool(int(x))
    if issubclass(typ, np.integer):
        return lambda x: int(float(x))
    elif issubclass(typ, np.floating):
        return float
    elif issubclass(typ, np.complex):
        return complex
    else:
        return str
def _string_like(obj):
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1

def savetxthd(fname, X, fmt='%.18e',delimiter=' ',header=None):
    """
    Save the data in X to file fname using fmt string to convert the
    data to strings

    Parameters
    ----------
    fname : filename or a file handle
        If the filename ends in .gz, the file is automatically saved in
        compressed gzip format.  The load() command understands gzipped
        files transparently.
    X : array or sequence
        Data to write to file.
    fmt : string or sequence of strings
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case delimiter is ignored.
    delimiter : str
        Character separating columns.
    header : str 
        A string which will be printed as a header before the data, 
        it is not '\n'-terminated 


    Examples
    --------
    >>> np.savetxt('test.out', x, delimiter=',') # X is an array
    >>> np.savetxt('test.out', (x,y,z)) # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e') # use exponential notation

    Notes on fmt
    ------------
    flags:
        - : left justify
        + : Forces to preceed result with + or -.
        0 : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated.

    precision:
        - For integer specifiers (eg. d,i,o,x), the minimum number of
          digits.
        - For e, E and f specifiers, the number of digits to print
          after the decimal point.
        - For g and G, the maximum number of significant digits.
        - For s, the maximum number of charac ters.

    specifiers:
        c : character
        d or i : signed decimal integer
        e or E : scientific notation with e or E.
        f : decimal floating point
        g,G : use the shorter of e,E or f
        o : signed octal
        s : string of characters
        u : unsigned decimal integer
        x,X : unsigned hexadecimal integer

    This is not an exhaustive specification.

    """

    if _string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname,'wb')
        else:
            fh = file(fname,'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')

    X = np.asarray(X)

    # Handle 1-dimensional arrays
    if X.ndim == 1:
        # Common case -- 1d array of numbers
        if X.dtype.names is None:
            X = np.atleast_2d(X).T
            ncol = 1

        # Complex dtype -- each field indicates a separate column
        else:
            ncol = len(X.dtype.descr)
    else:
        ncol = X.shape[1]

    # `fmt` can be a string with multiple insertion points or a list of formats.
    # E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
    if type(fmt) in (list, tuple):
        if len(fmt) != ncol:
            raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
        format = delimiter.join(fmt)
    elif type(fmt) is str:
        if fmt.count('%') == 1:
            fmt = [fmt,]*ncol
            format = delimiter.join(fmt)
        elif fmt.count('%') != ncol:
            raise AttributeError('fmt has wrong number of %% formats.  %s'
                                 % fmt)
        else:
            format = fmt

    if header != None:
        fh.write(header)
        
    for row in X:
        fh.write(format % tuple(row) + '\n')
