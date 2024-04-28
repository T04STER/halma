// Utilities for operations on numpy board
#pragma once
#include "Python.h"
#include <numpy/arrayobject.h>
#include <utility>

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


/*
  If given player (1) is in oposite camp return 1
  If given player (2) is in opostie camp return -1
  If player is not in camp return 0
*/
int player_in_enemy_camp(std::pair<int, int> pos, int player) {
  int pos_x = pos.first;
  int pos_y = pos.second;

  if (player == -1) {
    // check player 1 camp
    if ((pos_x > 4)
        || (pos_y > 4 )
        || (pos_x == 4 && pos_y > 1)
        || (pos_x == 3 && pos_y > 2) 
        || (pos_x == 2 && pos_y > 3)
    ) 
      return 0;
    return -1;
  }
  else {
    if ((pos_x < 11)
        || (pos_y < 11)
        || (pos_x == 11 && pos_y < 14)
        || (pos_x == 12 && pos_y < 13) 
        || (pos_x == 13 && pos_y < 12)
    ) 
      return 0;
    return 1;
  }
}