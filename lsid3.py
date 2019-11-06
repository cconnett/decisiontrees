# python3
# pylint: disable=g-complex-comprehension
"""Lookahead-by-Stochastic ID3 (LSID3)."""

import enum
import itertools
import math
import typing
from typing import (
    List,
    Mapping,
    Optional,
)

ROWS = 8
COLS = 8


class Coordinate(typing.NamedTuple):
  row: int
  col: int

  def __str__(self):
    return f'{chr(ord("A") + self.col)}{self.row + 1}'


class Board(object):
  pass


Universe = List[Board]


class Outcome(enum.Enum):
  MISS = 0
  HIT = 1
  SINK = 2


def Test(board, coord):
  return board[coord.row][coord.col]  # TODO(cjc): Detect SINK.


class Node(typing.NamedTuple):
  shot_taken: Coordinate
  parent: Optional['Node']
  universe: Universe
  # oneof {
  leaf_board: Board
  children: Mapping[Outcome, 'Node']
  # }


def ExpandNode(n: Node, coord: Coordinate):
  """Return node `n` expanded by splitting on `coord`."""
  splits = [[], [], []]
  for board in n.universe:
    splits[Test(board, coord)].append(board)
  if sum(bool(s) for s in splits) == 1:
    if sum(len(s) for s in splits) > 1:
      raise ValueError("This value can't split the set.")
    return n._replace(
        shot_taken=coord,
        children=None,
        leaf_board=next(itertools.chain.from_iterable(splits)),
    )
  return n._replace(
      shot_taken=coord,
      children=splits,
      leaf_board=None,
  )


def Lsid3(universe):
  pass


def ShannonEntropy(n: Node):
  """Return the entropy among the `children` of `n`."""
  return -sum(
      math.log2(len(child.universe) / len(n.universe))
      for child in n.children.values())


def EntropyK(k: int, n: Node):
  if k == 0:
    return ShannonEntropy(n)
  return


def ChooseAttribute(node: Node):
  domain = [Coordinate(i, j) for i in range(ROWS) for j in range(COLS)]
  induced_nodes = {coord: ExpandNode(node.universe, coord) for coord in domain}
  attribute_entropies = {
      coord: ShannonEntropy(induced_nodes[coord]) for coord in domain
  }
  return max(attribute_entropies.items(), key=lambda i: i[1])[0]


def ChooseAttributeK(k: int, node: Node):
  """Return a version of `node` split on the greatest expected entropy."""
  if k == 0:
    return ChooseAttribute(node)
  domain = [Coordinate(i, j) for i in range(ROWS) for j in range(COLS)]
  # TODO(cjc): Filter out already taken shots —or, because they would fail to
  # split the space, we'll get the ValueError from ExpandNode.
  induced_nodes = {coord: ExpandNode(node.universe, coord) for coord in domain}
  attribute_entropies = {
      coord: EntropyK(k - 1, induced_nodes[coord]) for coord in domain
  }
  return max(attribute_entropies.items(), key=lambda i: i[1])[0]
