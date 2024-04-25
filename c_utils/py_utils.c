// Utilities for simplyfying opperation such as adding to list etc.
#pragma once
#include "Python.h"
#include <numpy/arrayobject.h>


static void append_list(PyObject *listp, const int &x, const int &y) {
  PyObject *tuplep = PyTuple_New(2);
  PyObject *x_obj = PyLong_FromLong((long)x);
  PyObject *y_obj = PyLong_FromLong((long)y);
  PyTuple_SetItem(tuplep, 0, x_obj);
  PyTuple_SetItem(tuplep, 1, y_obj);
  PyList_Append(listp, tuplep);
}
