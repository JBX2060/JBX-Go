import numpy as np
from itertools import chain

class _Group:
    """Represent a solidly-connected group.

    Public attributes:
      colour
      points
      is_surrounded

    Points are coordinate pairs (row, col).

    """

class _Region:
    """Represent an empty region.

    Public attributes:
      points
      neighbouring_colours

    Points are coordinate pairs (row, col).

    """
    def __init__(self):
        self.points = set()
        self.neighbouring_colours = set()

class Board:
    """A legal Go position.

    Supports playing stones with captures, and area scoring.

    Public attributes:
      side         -- board size (int >= 2)
      board_points -- list of coordinates of all points on the board

    """
    def __init__(self, side):
        self.side = side
        if side < 2:
            raise ValueError
        self.board_points = np.array(np.meshgrid(np.arange(side), np.arange(side))).T.reshape(-1, 2)
        self.board = np.full((side, side), None)
        self._is_empty = True
        self._group_cache = {}
        self._region_cache = {}

    def copy(self):
        """Return an independent copy of this Board."""
        b = Board(self.side)
        b.board = np.copy(self.board)
        b._is_empty = self._is_empty
        return b

    def _process_points(self, row, col, process_function):
        points = set()
        to_handle = [np.array([row, col])]

        while to_handle:
            point = to_handle.pop()
            points.add(tuple(point))
            r, c = point
            neighbours = np.array([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            valid_neighbours = neighbours[np.logical_and(np.all(neighbours >= 0, axis=1), 
                                                         np.all(neighbours < self.side, axis=1))]
            for neighbour in valid_neighbours:
                process_function(point, neighbour, points, to_handle)

        return points

    def _make_group(self, row, col, colour):
        point = (row, col)
        if point in self._group_cache:
            return self._group_cache[point]

        is_surrounded = [True]

        def process_group_points(point, neighbour, points, to_handle):
            r1, c1 = neighbour
            neigh_colour = self.board[r1][c1]
            if neigh_colour is None:
                is_surrounded[0] = False
            elif neigh_colour == colour:
                if tuple(neighbour) not in points:
                    to_handle.append(neighbour)

        points = self._process_points(row, col, process_group_points)
        group = _Group()
        group.colour = colour
        group.points = points
        group.is_surrounded = is_surrounded[0]
        for p in points:
            self._group_cache[p] = group
        return group

    def _make_empty_region(self, row, col):
        point = (row, col)
        if point in self._region_cache:
            return self._region_cache[point]

        neighbouring_colours = set()

        def process_region_points(point, neighbour, points, to_handle):
            r1, c1 = neighbour
            neigh_colour = self.board[r1][c1]
            if neigh_colour is None:
                if tuple(neighbour) not in points:
                    to_handle.append(neighbour)
            else:
                neighbouring_colours.add(neigh_colour)

        points = self._process_points(row, col, process_region_points)
        region = _Region()
        region.points = points
        region.neighbouring_colours = neighbouring_colours
        for p in points:
            self._region_cache[p] = region
        return region

    def _find_surrounded_groups(self, r, c):
        """Find solidly-connected groups with 0 liberties adjacent to r,c.

        Returns a list of _Groups.

        """
        surrounded = []
        handled = set()
        for (row, col) in [(r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if not ((0 <= row < self.side) and (0 <= col < self.side)):
                continue

            colour = self.board[row][col]
            if colour is None:
                continue

            point = (row, col)
            if point in handled:
                continue

            group = self._make_group(row, col, colour)
            if group.is_surrounded:
                surrounded.append(group)
            handled.update(group.points)
        return surrounded

    def _find_all_surrounded_groups(self):
        """Find all solidly-connected groups with 0 liberties.

        Returns a list of _Groups.

        """
        return self._find_surrounded_groups()
    
    def is_empty(self):
        return self._is_empty

    def get(self, row, col):
        if row < 0 or col < 0:
            raise IndexError
        return self.board[row, col]

    def play(self, row, col, colour):
        if row < 0 or col < 0:
            raise IndexError
        opponent = 1 if colour == -1 else -1
        if self.board[row, col] is not None:
            raise ValueError
        self.board[row, col] = colour
        self._is_empty = False
        surrounded = self._find_surrounded_groups(row, col)
        simple_ko_point = None

        if surrounded:
            surrounded_len = [len(group.points) for group in surrounded]
            if len(surrounded) == 1 and np.sum(surrounded_len) == self.side * self.side:
                self._is_empty = True
            else:
                to_capture = [group for group in surrounded if group.colour == opponent]
                if len(to_capture) == 1 and len(to_capture[0].points) == 1:
                    self_capture = [group for group in surrounded if group.colour == colour]
                    if len(self_capture[0].points) == 1:
                        simple_ko_point = list(to_capture[0].points)[0]
            for group in to_capture:
                for r, c in group.points:
                    self.board[r, c] = None
        return simple_ko_point

    def apply_setup(self, black_points, white_points, empty_points):
        for (row, col) in chain(black_points, white_points, empty_points):
            if row < 0 or col < 0 or row >= self.side or col >= self.side:
                raise IndexError
        self.board[tuple(zip(*black_points))] = 'b'
        self.board[tuple(zip(*white_points))] = 'w'
        self.board[tuple(zip(*empty_points))] = None
        captured = self._find_all_surrounded_groups()
        for group in captured:
            self.board[tuple(zip(*group.points))] = None
        self._is_empty = np.all(self.board == None)
        return not(captured)

    def list_occupied_points(self):
        occupied_points = np.argwhere(self.board != None)
        colours = self.board[self.board != None]
        return list(zip(colours, [tuple(point) for point in occupied_points]))

    def area_score(self):
        scores = {'b': 0, 'w': 0}
        handled = set()

        for row, col in self.board_points:
            colour = self.board[row, col]
            if colour is not None:
                scores[colour] += 1
                continue
            point = (row, col)
            if point in handled:
                continue
            region = self._make_empty_region(row, col)
            region_size = len(region.points)
            for colour in ('b', 'w'):
                if colour in region.neighbouring_colours:
                    scores[colour] += region_size
            handled.update(region.points)
        return scores['b'] - scores['w']

