#pragma once
#include "Python.h"
#include <numpy/arrayobject.h>
#include "py_utils.c"
#include "board_utils.c"
#include <vector>
#include <stack>
#include <set>
#include <iostream>

#define NUM_NEIGHBOURS  8

std::vector<std::pair<int, int>> find_next_jumps(PyArrayObject *boardp, const std::pair<int, int> &pos, const int &len) {
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

  std::vector<std::pair<int, int>> possible_jumps;
  for (int i = 0; i < NUM_NEIGHBOURS; i++) {
    auto jump_over = neighbours[i];
    auto jump_dest = jumps[i];
    if (
      is_on_board(jump_over.first, jump_over.second, len) 
      && !is_empty(boardp, jump_over.first, jump_over.second)
      && is_on_board(jump_dest.first, jump_dest.second, len)
      && is_empty(boardp, jump_dest.first, jump_dest.second)
    ) {
        possible_jumps.push_back(jump_dest);
    }
  }
  return possible_jumps;
}


static void dfs_jump_search(PyArrayObject *boardp, PyObject *listp, const std::pair<int, int> &pos) {
  std::stack<std::pair<int, int>> stack;
  std::set<std::pair<int, int>> visited;
  stack.push(pos);
  while (!stack.empty()) {
    std::pair<int, int> current = stack.top();
    stack.pop();

    if (visited.find(current) == visited.end()) {
      visited.insert(current);
      append_list(listp, current.first, current.second);
      auto next_jumps = find_next_jumps(boardp, current, 16);
      for (const auto &jump : next_jumps) {
        stack.push(jump);
      }
    }
  }

}

static void dfs_jump_search(PyArrayObject *boardp, std::set<std::pair<int, int>> *visitedp, const std::pair<int, int> &pos) {
  std::stack<std::pair<int, int>> stack;
  stack.push(pos);
  while (!stack.empty()) {
    std::pair<int, int> current = stack.top();
    stack.pop();

    if (visitedp->find(current) == visitedp->end()) {
      visitedp->insert(current);
      auto next_jumps = find_next_jumps(boardp, current, 16);
      for (const auto &jump : next_jumps) {
        stack.push(jump);
      }
    }
  }

}



static PyObject* jump_moves(PyObject* self, PyObject* args){
  PyArrayObject *boardp;
  PyObject *positionp;
  PyObject *list = PyList_New(0);


  if (!PyArg_ParseTuple(args,"O!O", &PyArray_Type, &boardp, &positionp))
      return list;

  int len = get_dimensions_len(boardp);
  if (len == -1 ){
    return list;
  }
  int pos_x, pos_y;
  if (!PyArg_ParseTuple(positionp, "ii", &pos_x, &pos_y))
      return list;
  auto va = (int)array_get(boardp, pos_x, pos_y);
  //std::cout<< "(" << pos_x<< ", " << pos_y << ") given val " << va << " \n";

  if (!is_on_board(pos_x, pos_y, len) || !is_empty(boardp, pos_x, pos_y)) {
    return list;
  }

  std::pair<int, int> pos_tup = {pos_x, pos_y};
  dfs_jump_search(boardp, list, pos_tup);

  return list;
}
