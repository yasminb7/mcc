Installation
====================

There is no need to install *mcc* in the proper sense of the term.
You can simply copy the mcc folder to anywhere you like and start working with it.
However, there are a few packages you need to install before running *mcc*.


Dependencies
------------
`mcc` is written in `Python <http://www.python.org/>`_ (version 2.7.3) and uses the following libraries:

* `NumPy <http://www.numpy.org/>`_ (version 1.6.1)
* `SciPy <http://www.scipy.org/>`_ (version 0.9.0)
* `matplotlib <http://matplotlib.org/>`_ (version 1.1.1)

You don't need to have these exact versions. If you have newer versions, they should be fine too. However, don't try to run the code with Python 3.x, it won't work. Stick to Python.2.7.x.

Some parts that slowed down the execution of the program were rewritten in C. The ``scipy.weave`` package was used to execute C code from python.
Depending on how you got the source code for *mcc*, it could be that you don't need a C compiler.
If you modify the parts written in C, you will need to recompile them for the program to run correctly.
Refer to the documentation for the ``scipy.weave`` package (you can find it online) to find out how you can use your compiler to recompile the code.

If you're using Linux, I assume you know how to find and install these packages. If on top of that you have a working installation of ``gcc`` (the GNU C Compiler), you should be all set.

If you're using Windows, I recommend installing `WinPython <http://code.google.com/p/winpython/>`_. Make sure you get a version for Python 2.7.x, it won't work otherwise. WinPython contains all the libraries mentioned above. The only thing you're still missing is the C compiler, but I'll have to let you figure that out because it all depends on your particular system. Again, refer to the documentation for the ``scipy.weave`` package.

Using a C/C++ compiler on Windows
---------------------------------
If you want to compile extensions for Python in C or C++ on a 64bit Windows (in particular using scipy.weave) you need to install these two products:

Install Microsoft Visual C++ 2008 Express
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You need Visual C++ 2008 Express to compile C/C++ libraries to work with Python 2.7.x on Windows.
The installer is in the file `vcsetup.exe`.

Install Windows 7 SDK
^^^^^^^^^^^^^^^^^^^^^
In addition, you need to install the Windows 7 SDK because it contains the 64bit tools needed.
There are tools to mount an iso-file in Windows like a drive.
Google "mount iso windows" if you want to know more.
The installer you want to execute is `setup.exe` from the file `GRMSDKX_EN_DVD.iso`.

After you've installed it go to the Visual C++ 2008 configuration program that lets you choose which SDK to use, and pick the Windows 7 SDK you just installed.

Compiling
^^^^^^^^^
There is no need to separately compile the code before you run it. If you launch

>>> python main.py

`scipy.weave` will check if the C code has been modified and will recompile it if needed.
In order to have the right environment variables set (useful for letting `weave` know which compiler to use etc.), you should launch *mcc* with the command above **from the CMD-Shell coming with Windows 7 SDK**.

