from distutils.core import setup, Extension

module1 = Extension('cgeometry',
		    sources=['geometry.c'])

setup(name = 'Pakcagename',
      version='1.0',
      description='Python extension of geomplane',
      ext_modules=[module1])

