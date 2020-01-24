# python3
# pylint: disable=g-complex-comprehension
"""Lookahead-by-Stochastic ID3 (LSID3)."""

import collections
import enum
import itertools
import math
import pickle
import random
import sys

from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

ROWS = 5
COLS = 5


class Coordinate(NamedTuple):
  row: int
  col: int

  def __repr__(self):
    return f'{chr(ord("A") + self.col)}{self.row + 1}'


Attribute = Coordinate


class Direction(enum.Enum):
  HORIZONTAL = 0
  VERTICAL = 1


class Outcome(enum.Enum):
  MISS = 0
  HIT = 1
  SINK = 2


class Board(object):
  """A battleships board."""
  ships: List[Tuple[Coordinate, Direction, int]]
  occupied: Set[Coordinate]

  def __init__(self, ships):
    self.ships = ships
    self.occupied = set()
    self.valid = True
    for start, direction, length in self.ships:
      r, c = start.row, start.col
      for _ in range(length):
        if not (0 <= r < ROWS and 0 <= c < COLS):
          self.valid = False
          return
        now = Coordinate(r, c)
        if now in self.occupied:
          self.valid = False
          return
        self.occupied.add(now)
        if direction == Direction.HORIZONTAL:
          c += 1
        else:
          r += 1

  def Test(self, coord) -> Outcome:
    if coord not in self.occupied:
      return Outcome.MISS
    return Outcome.HIT  # TODO(cjc): Detect SINK.


def GetShips(length):
  for row in range(ROWS):
    for col in range(COLS):
      for direction in Direction:
        yield (Coordinate(row, col), direction, length)


def GetAllBoards():
  ox = {}
  for two, three, four in itertools.product(
      GetShips(2), GetShips(3), GetShips(4)):
    board = Board([two, three, four])
    if board.valid:
      ox[tuple(sorted(board.occupied))] = board
  return ox.values()


Universe = List[Board]
AttributeSelector = Callable[['Node', Iterable[Attribute]], Attribute]


class Node(object):
  """A decision tree node."""
  attribute: Optional[Attribute]
  parent: Optional['Node']
  universe: Universe
  previous_shots: Set[Attribute]
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

    self.previous_shots = set()
    while par:
      par = par.parent
      if par:
        self.previous_shots.add(par.attribute)

  def String(self, depth=1, indent=0):
    """Fancy string."""
    if depth <= 0:
      return str(len(self.universe))
    head = f'{len(self.universe):4d}: {self.attribute} → '
    indent += len(head)

    if depth == 1:
      sublist = ', '.join(s.String(0, indent) for s in self.children.values())
      return head + f'[{sublist}]'
    return head + f'\n{" " * indent}'.join(
        s.String(depth - 1, indent) for s in self.children.values())

  def __str__(self):
    return self.String(4)

  def Expand(self, attr: Attribute) -> None:
    """Expand node `n` in-place by splitting on `attr`."""
    if len(self.universe) == 1:
      self.attribute = None
      self.children = None
      self.leaf = self.universe[0]
      return

    self.attribute = attr
    splits = collections.defaultdict(list)
    for element in self.universe:
      splits[element.Test(attr)].append(element)
    self.children = {
        outcome: Node(None, self, split, None, None)
        for outcome, split in splits.items()
    }
    self.leaf = None

  def ExpandTree(self, attributes: Set[Attribute],
                 choose_attribute: AttributeSelector) -> float:
    """Expand `root` in place, using `attributes`."""
    if len(self.universe) == 1:
      return int(len(self.previous_shots) < 16)

    best_attr = choose_attribute(self, attributes)
    self.Expand(best_attr)
    if len(self.previous_shots) >= 16:
      return 0
    attributes_prime = attributes - {best_attr}
    level = 25 - len(attributes)
    s = 0
    for outcome, child in self.children.items():
      if level == 7 and 'L' in choose_attribute.__name__:
        pedigree = [outcome.value]
        par = self.parent
        ref = self
        while par:
          for o in par.children:
            if par.children[o] is ref:
              pedigree.append(o.value)
          ref = par
          par = par.parent
        byte = 0
        for bit in pedigree[::-1]:
          byte <<= 1
          byte |= bit
        print(byte)
      s += child.ExpandTree(attributes_prime, choose_attribute)
    return s


def I(seq):
  """Shannon's information metric."""
  seq = list(seq)
  total = sum(seq)
  return -sum(ci / total * math.log2(ci / total) for ci in seq if ci)


def MaxEntropy(n) -> float:
  return math.log2(n)


def GainK(n: Node, attributes: Set[Attribute], cur_attr: Attribute, k: int):
  return MaxEntropy(len(n.universe)) - EntropyK(n, attributes, cur_attr, k)


def EntropyK(n: Node, attributes: Set[Attribute], cur_attr: Attribute, k: int):
  """Entropy after looking k nodes deep."""
  n.Expand(cur_attr)
  if n.leaf:
    return 0.0

  weighted_sum = 0.0
  for subnode in n.children.values():
    if k == 1:
      # Shortcut getting the same result for each next_attr (which we don't
      # split on).
      best_entropy = MaxEntropy(len(subnode.universe))
    else:
      best_entropy = min(
          EntropyK(subnode, attributes - {next_attr}, next_attr, k - 1)
          for next_attr in attributes)
    weighted_sum += len(subnode.universe) / len(n.universe) * best_entropy
  return weighted_sum


def SID3ChooseAttribute(root: Node,
                        attributes: Iterable[Attribute]) -> Attribute:
  max_entropy = MaxEntropy(len(root.universe))
  gains = {a: GainK(root, attributes, a, 1) for a in attributes}
  perfects = [a for a in gains if math.isclose(max_entropy, gains[a])]
  if perfects:
    return random.choice(perfects)
  else:
    return random.choices(list(gains.keys()), list(gains.values()))[0]


def MakeLSID3ChooseAttribute(r: int) -> AttributeSelector:
  """Make a chooser with param `r`."""

  def LSID3ChooseAttribute(root: Node,
                           attributes: Iterable[Attribute]) -> Attribute:
    """Stochastically build trees and count wins."""
    best, best_a = -1, None
    for i, a in enumerate(attributes):
      total_winning_lines = 0
      root.Expand(a)
      level = 25 - len(attributes)
      if level < 7:
        print(f'Expanding tree at level {level} ({i+1} of {len(attributes)}).')
      for child in root.children.values():
        total_winning_lines = (
            child.ExpandTree(attributes - {a}, SID3ChooseAttribute))
        # total_winning_lines += max(
        #
        #     for _ in range(r))
      if total_winning_lines > best:
        best = total_winning_lines
        best_a = a
    if level < 7:
      print(f'LSID3-Choose-Attribute chose {best_a} with {best} winning lines.')
    return best_a

  return LSID3ChooseAttribute


def MakeID3kChooseAttribute(k: int) -> AttributeSelector:

  def ID3kChooseAttribute(root: Node,
                          attributes: Iterable[Attribute]) -> Attribute:
    gains = {a: EntropyK(root, attributes, a, k) for a in attributes}
    return max(gains, key=lambda a: gains[a])

  return ID3kChooseAttribute


def main(_):
  domain = {Coordinate(i, j) for i in range(ROWS) for j in range(COLS)}
  # pickle.dump(list(GetAllBoards()), open('boards', 'wb'))
  # b = pickle.load(open('boards', 'rb'))
  b = list(GetAllBoards())
  t = Node(None, None, b, None, None)
  # t.ExpandTree(domain, MakeID3kChooseAttribute(2))
  try:
    t.ExpandTree(domain, MakeLSID3ChooseAttribute(1))
  except:
    import pdb
    pdb.post_mortem()
  print(t)


if __name__ == '__main__':
  main(sys.argv)
