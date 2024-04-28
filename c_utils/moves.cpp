#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#pragma once
#include "Python.h"
#include <numpy/arrayobject.h>
#include "py_utils.c"
#include "board_utils.cpp"
#include "jumps.cpp"
#include <set>
#include <iostream>


struct NeighboursOrJumpsTuple{
  // neighbours & possible jumps
  std::vector<std::pair<int, int>> neighbour;
  std::vector<std::pair<int, int>> jumps_destinations;
};


NeighboursOrJumpsTuple get_neighbours(PyArrayObject *boardp, const std::pair<int, int> &pos, const int& len, const int& player_flag_camp) {
  int x = pos.first, y = pos.second; 
  int xt = x + 1, xt2 = x + 2;
  int xb = x - 1, xb2 = x - 2;
  int yl = y - 1, yl2 = y - 2;
  int yr = y + 1, yr2 = y + 2;
    
  std::pair<int, int> neighbours[NUM_NEIGHBOURS] = {
        {xt, yl}, {xt, y}, {xt, yr},
        {x, yl}, {x, yr},
        {xb, yl}, {xb, y}, {xb, yr}
  };
  std::pair<int,int> jumps[NUM_NEIGHBOURS]= {
    {xt2, yl2}, {xt2, y}, {xt2, yr2},
    {x, yl2}, {x, yr2},
    {xb2, yl2}, {xb2, y}, {xb2, yr2}
  };
    
  std::vector<std::pair<int, int>> valid_neighbours;
  std::vector<std::pair<int, int>> possible_jumps;
  for (int i = 0; i < NUM_NEIGHBOURS; i++) {
    auto neighbour = neighbours[i];
    if (is_on_board(neighbour.first, neighbour.second, len) ) {
      if (is_empty(boardp, neighbour.first, neighbour.second)) {
        if (player_flag_camp == 0 || player_in_enemy_camp(neighbour, player_flag_camp) == player_flag_camp)
          valid_neighbours.push_back(neighbour);
      }
      else {
        auto jump_dest = jumps[i];
        if (is_on_board(jump_dest.first, jump_dest.second, len) && is_empty(boardp, jump_dest.first, jump_dest.second)) {
          if (player_flag_camp == 0 || player_in_enemy_camp(jump_dest, player_flag_camp) == player_flag_camp)
            possible_jumps.push_back(jump_dest);
        }
      }
    }
  }
  NeighboursOrJumpsTuple result = {valid_neighbours, possible_jumps};
  return result;
}



static PyObject* get_pawn_moves(PyObject* self, PyObject* args) {
  PyArrayObject *boardp;
  PyObject *positionp;
  PyObject *list = PyList_New(0);

  if (!PyArg_ParseTuple(args,"O!O", &PyArray_Type, &boardp, &positionp))
      return list;

  int pos_x, pos_y;
  if (!PyArg_ParseTuple(positionp, "ii", &pos_x, &pos_y))
    return list;

  const int len = get_dimensions_len(boardp);



  if (!is_on_board(pos_x, pos_y, len) || is_empty(boardp, pos_x, pos_y)) {
    return list;
  }
  const std::pair<int, int> pos_tup = {pos_x, pos_y};
  int player = array_get(boardp, pos_x, pos_y);
  int player_in_camp_flag = player_in_enemy_camp(pos_tup, player);

  NeighboursOrJumpsTuple jumps_neighbour = get_neighbours(boardp, pos_tup, len, player_in_camp_flag);
  std::set<std::pair<int, int>> visited;
  for (const auto &neighbour: jumps_neighbour.neighbour) {
    visited.insert(neighbour);
  }
  
  for (const auto &jump: jumps_neighbour.jumps_destinations) {
    dfs_jump_search(boardp, &visited, jump, player_in_camp_flag);
  }

  for (const auto &pair : visited) {
    append_list(list, pair.first, pair.second);
  }
  
  return list;
}

static PyMethodDef mainMethods[] = {
    {"jump_moves", jump_moves, METH_VARARGS, "Evaluate jump moves"},
    {"get_pawn_moves", get_pawn_moves, METH_VARARGS, "Generate all possible moves"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef movesModule = {
    PyModuleDef_HEAD_INIT,
    "moves",
    "Evaluate jump moves from given position",
    -1,
    mainMethods
};

PyMODINIT_FUNC PyInit_moves(void){
    import_array();
    return PyModule_Create(&movesModule);
}