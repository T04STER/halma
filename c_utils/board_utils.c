// Utilities for operations on numpy board
#pragma once
#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdbool.h> 

static int array_get (PyArrayObject *boardp, int x, int y) {
  // wraper for getting an integer
  npy_int8 *value_p = (npy_int8 *) PyArray_GETPTR2(boardp, x, y);
  return (int)*value_p;
}


static bool is_on_board(int x, int y, int len) {
  return 0 <= x && x < len && 0 <= y && y < len;
}

static bool is_empty(PyArrayObject *boardp, int x, int y) {
  return array_get(boardp, x, y) == 0;
}

static int get_dimensions_len(PyArrayObject *boardp) {
  if (PyArray_NDIM(boardp) != 2) {
      PyErr_SetString(PyExc_ValueError, "Input array must be 2D");
      return -1;
  }
  npy_intp *dimensions = PyArray_DIMS(boardp);
  int rows = (int)dimensions[0];
  int cols = (int)dimensions[1];
  
  if (rows != cols) {
    char error_message[100];
    sprintf(error_message, "Board must be a matrix, is (%d, %d)", rows, cols);
    PyErr_SetString(PyExc_ValueError, error_message);
      return -1;
  }
  return rows;
}