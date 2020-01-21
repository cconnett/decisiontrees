# python3
# pylint: disable=g-complex-comprehension
"""Lookahead-by-Stochastic ID3 (LSID3)."""

import collections
import enum
import itertools
import math
import pickle
import sys
from typing import (
    Collection,
    NamedTuple,
    Set,
    List,
    Mapping,
    Optional,
    Tuple,
)

ROWS = 8
COLS = 8
K = 1


class Coordinate(NamedTuple):
  row: int
  col: int

  def __str__(self):
    return f'{chr(ord("A") + self.col)}{self.row + 1}'


class Direction(enum.Enum):
  HORIZONTAL = 0
  VERTICAL = 1


class Board(NamedTuple):
  ships: List[Tuple[Coordinate, Direction, int]]

  @property
  def occupied(self):
    for start, direction, length in self.ships:
      current = start
      attr = 'row' if direction == Direction.HORIZONTAL else 'col'
      for _ in range(length):
        if 0 <= current.row < ROWS and 0 <= current.col < COLS:
          yield current
        current = current._replace(**{attr: getattr(current, attr) + 1})


def GetShips(length):
  for row in range(ROWS):
    for col in range(COLS):
      for direction in Direction:
        yield (Coordinate(row, col), direction, length)


def GetAllBoards():
  for two, three, four in itertools.product(
      GetShips(2), GetShips(3), GetShips(4)):
    board = Board([two, three, four])
    if len(set(board.occupied)) == 9:
      yield board


Universe = List[Board]
Attribute = Coordinate


class Outcome(enum.Enum):
  MISS = 0
  HIT = 1
  SINK = 2


def Test(board, coord) -> Outcome:
  if coord not in board.occupied:
    return Outcome.MISS
  return Outcome.HIT  # TODO(cjc): Detect SINK.


class Node(NamedTuple):
  shot_taken: Optional[Coordinate]
  parent: Optional['Node']
  universe: Universe
  # oneof {
  leaf_board: Board
  children: Mapping[Outcome, 'Node']
  # }


def ExpandNode(n: Node, attr: Attribute) -> Node:
  """Return node `n` expanded by splitting on `attr`."""
  splits = collections.defaultdict(list)
  for element in n.universe:
    splits[Test(element, attr)].append(element)
  if sum(bool(s) for s in splits) == 1:
    return n._replace(
        shot_taken=attr,
        children=None,
        leaf_board=next(itertools.chain.from_iterable(splits)),
    )
  return n._replace(
      shot_taken=attr,
      children={
          outcome: Node(None, n, split, None, None)
          for outcome, split in splits.items()
      },
      leaf_board=None,
  )


def TotalEntropy(n: Node) -> float:
  """Return the entropy among the `children` of `n`."""
  if n.leaf_board:
    return 0.0
  return -(math.log2(1 / len(n.universe)) * len(n.universe))


def GainK(n: Node, attributes: Collection[Attribute], cur_attr: Attribute,
          k: int) -> float:
  return TotalEntropy(n) - EntropyK(n, attributes, cur_attr, k)


def EntropyK(n: Node, attributes: Set[Attribute], cur_attr: Attribute, k: int):
  """Entropy after looking k nodes deep."""
  if k == 0:
    return TotalEntropy(n)

  expanded_node = ExpandNode(n, cur_attr)
  weighted_sum = 0
  for subnode in expanded_node.children.values():
    weighted_sum += (
        len(subnode.universe) / len(n.universe) * min(
            EntropyK(subnode, attributes - {next_attr}, next_attr, k - 1)
            for next_attr in attributes))
  return weighted_sum


def ExpandTree(root: Node, attributes: Set[Attribute]):
  best_attr = min(attributes, key=lambda a: EntropyK(root, attributes, a, K))
  new_root = ExpandNode(root, best_attr)
  attributes_prime = attributes - {best_attr}
  return new_root._replace(children={
      v: ExpandTree(n, attributes_prime) for v, n in new_root.children.items()
  })


def main(_):
  domain = {Coordinate(i, j) for i in range(ROWS) for j in range(COLS)}
  b = pickle.load(open('boards', 'rb'))
  t = Node(None, None, b, None, None)
  e = ExpandTree(t, domain)
  print(e)


# def ChooseAttribute(node: Node):
#
#   induced_nodes = {coord: ExpandNode(node.universe, coord) for coord in domain}
#   attribute_entropies = {
#       coord: ShannonEntropy(induced_nodes[coord]) for coord in domain
#   }
#   return max(attribute_entropies.items(), key=lambda i: i[1])[0]

# def ChooseAttributeK(k: int, node: Node):
#   """Return a version of `node` split on the greatest expected entropy."""
#   if k == 0:
#     return ChooseAttribute(node)
#   domain = [Coordinate(i, j) for i in range(ROWS) for j in range(COLS)]
#   # TODO(cjc): Filter out already taken shots —or, because they would fail to
#   # split the space, we'll get the ValueError from ExpandNode.
#   induced_nodes = {coord: ExpandNode(node.universe, coord) for coord in domain}
#   attribute_entropies = {
#       coord: EntropyK(k - 1, induced_nodes[coord]) for coord in domain
#   }
#   return max(attribute_entropies.items(), key=lambda i: i[1])[0]

if __name__ == '__main__':
  main(sys.argv)
