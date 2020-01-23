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

ROWS = 5
COLS = 5
K = 2


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


class Node(object):
  """A decision tree node."""
  attribute: Optional[Coordinate]
  parent: Optional['Node']
  universe: Universe
  # oneof {
  leaf: Board
  children: Mapping[Outcome, 'Node']

  # }

  def __init__(self, attr, par, uni, leaf, childs):
    self.attribute = attr
    self.parent = par
    self.universe = uni
    self.leaf = leaf
    self.children = childs

  def String(self, depth=1):
    if depth <= 0:
      return str(len(self.universe))
    return (f'{self.attribute}: {len(self.universe)} → ' +
            f'[{", ".join(s.String(depth-1) for s in self.children.values())}]')

  def __str__(self):
    return self.String(2)


def ExpandNode(n: Node, attr: Attribute) -> None:
  """Expand node `n` in-place by splitting on `attr`."""
  splits = collections.defaultdict(list)
  for element in n.universe:
    splits[Test(element, attr)].append(element)
  n.attribute = attr
  if sum(bool(s) for s in splits) == 1:
    n.children = None
    n.leaf = next(itertools.chain.from_iterable(splits.values()))
  else:
    n.children = {
        outcome: Node(None, n, split, None, None)
        for outcome, split in splits.items()
    }
    n.leaf = None


def TotalEntropy(n: Node) -> float:
  """Return the entropy among the `children` of `n`."""
  if n.leaf:
    return 0.0
  return -(math.log2(1 / len(n.universe)) * len(n.universe))


def GainK(n: Node, attributes: Collection[Attribute], cur_attr: Attribute,
          k: int) -> float:
  return TotalEntropy(n) - EntropyK(n, attributes, cur_attr, k)


def EntropyK(n: Node, attributes: Set[Attribute], cur_attr: Attribute, k: int):
  """Entropy after looking k nodes deep."""
  if k == 0:
    return TotalEntropy(n)

  ExpandNode(n, cur_attr)
  if n.leaf:
    return 0.0
  weighted_sum = 0.0
  for subnode in n.children.values():
    weighted_sum += (
        len(subnode.universe) / len(n.universe) * min(
            EntropyK(subnode, attributes - {next_attr}, next_attr, k - 1)
            for next_attr in attributes))
  return weighted_sum


def ExpandTree(root: Node, attributes: Set[Attribute]):
  best_attr = max(attributes, key=lambda a: GainK(root, attributes, a, K))
  ExpandNode(root, best_attr)
  if root.children:
    attributes_prime = attributes - {best_attr}
    for n in root.children.values():
      ExpandTree(n, attributes_prime)


def main(_):
  domain = {Coordinate(i, j) for i in range(ROWS) for j in range(COLS)}
  # pickle.dump(list(GetAllBoards()), open('boards', 'wb'))
  b = pickle.load(open('boards', 'rb'))
  t = Node(None, None, b, None, None)
  ExpandTree(t, domain)
  print(t)


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
