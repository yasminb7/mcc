Installation
====================


`mcc` is written in `Python <http://www.python.org/>`_ (version 2.7.3) and uses the following libraries:

* `NumPy <http://www.numpy.org/>`_ (version 1.6.1)
* `SciPy <http://www.scipy.org/>`_ (version 0.9.0)
* `matplotlib <http://matplotlib.org/>`_ (version 1.1.1)

Some parts that slowed down the execution of the program were rewritten in C. Depending on how you got the source code for *mcc*, it could be that you don't need a C compiler.
If you modify the parts written in C, you will need to recompile them for the program to run correctly. Refer to the documentation for the ``scipy.weave`` package (you can find it online) to find out how you can use your compiler to recompile the code.

If you're using Linux, I assume you know how to find and install these packages. If on top of that you have a working installation of ``gcc`` (the GNU C Compiler), you should be all set.

If you're using Windows, I recommend installing `WinPython <http://code.google.com/p/winpython/>`_. Make sure you get a version for Python 2.7.x, it won't work otherwise. WinPython contains all the libraries mentioned above. The only thing you're still missing is the C compiler, but I'll have to let you figure that out because it all depends on your particular system. Again, refer to the documentation for the ``scipy.weave`` package.

