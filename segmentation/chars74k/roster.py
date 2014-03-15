# from collections import Counter

DELETION_COST = 1
INSERTION_COST = 1
CHANGE_COST = 1

LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class Roster:
  def __init__(self, filename):
    self.roster = []

    with open(filename, 'r') as handle:
      for line in handle:
        self.roster.append(line.strip().lower().replace(' ', ''))

  def get_closest_name(self, predicted_name):
    predicted_name = predicted_name.replace(' ', '')
    max_value = float('-Inf')
    closest_name = None
    gap_penalties = (0, 0, 0, 0)
    match_matrix = dict()
    for letter1 in LETTERS:
      for letter2 in LETTERS:
        match_matrix[(letter1, letter2)] = (1 if letter1 == letter2 else 0)

    for possible_name in self.roster:
      m_score_matrix, Ix_score_matrix, Iy_score_matrix = self.initialize_score_matrices(
        predicted_name, possible_name, gap_penalties)

      rows = len(possible_name) + 1
      cols = len(predicted_name) + 1

      for row in range(1, rows):
        for col in range(1, cols):
          self.update_scores(m_score_matrix, Ix_score_matrix, Iy_score_matrix, row, col,
            match_matrix, predicted_name[col-1], possible_name[row-1], gap_penalties)

      cur_max_value = self.find_max_value(m_score_matrix, Ix_score_matrix, Iy_score_matrix,
        rows, cols, predicted_name, possible_name)

      if max_value < cur_max_value:
        max_value = cur_max_value
        closest_name = possible_name

      return closest_name

  def initialize_score_matrices(self, predicted_name, possible_name, gap_penalties):
    rows = len(possible_name) + 1
    cols = len(predicted_name) + 1
    m_score_matrix = create_matrix(rows, cols)
    Ix_score_matrix = create_matrix(rows, cols)
    Iy_score_matrix = create_matrix(rows, cols)

    dx, ex, dy, ey = gap_penalties
    for r in range(rows):
      initial_gap_penalty = -1 * dx * r
      Ix_score_matrix[r][0] = initial_gap_penalty
      Iy_score_matrix[r][0] = initial_gap_penalty
    for c in range(cols):
      initial_gap_penalty = -1 * dy * c
      Ix_score_matrix[0][c] = initial_gap_penalty
      Iy_score_matrix[0][c] = initial_gap_penalty

    return (m_score_matrix, Ix_score_matrix, Iy_score_matrix)


  def update_scores(self, m_score_matrix, Ix_score_matrix, Iy_score_matrix, row, col,
                    match_matrix, predicted_name_letter, possible_name_letter, gap_penalties):
    # dx is gap open penalty of A, ex is gap extension penalty for A, dy is gap
    # open penalty of B, ey is gap extension penalty for B
    dx, ex, dy, ey = gap_penalties

    # Update the score matrices
    match_value = match_matrix[(predicted_name_letter, possible_name_letter)]
    m_score_matrix[row][col] = max(m_score_matrix[row-1][col-1] + match_value,
                                   Ix_score_matrix[row-1][col-1] + match_value,
                                   Iy_score_matrix[row-1][col-1] + match_value)
    Ix_score_matrix[row][col] = max(m_score_matrix[row][col-1] - dy,
                                    Ix_score_matrix[row][col-1] - ey)
    Iy_score_matrix[row][col] = max(m_score_matrix[row-1][col] - dx,
                                    Iy_score_matrix[row-1][col] - ex)


  def find_max_value(self, m_score_matrix, Ix_score_matrix, Iy_score_matrix,
                     rows, cols, predicted_name, possible_name):
    max_value = -1
    # Search the margins for the max
    for row in range(rows):
      if m_score_matrix[row][cols-1] > max_value:
        max_value = m_score_matrix[row][cols-1]
    for col in range(cols):
      if m_score_matrix[rows-1][col] > max_value:
        max_value = m_score_matrix[rows-1][col]

      # Output the maximum score
      return max_value


def create_matrix(rows, cols):
  matrix = []
  for i in range(rows):
    matrix.append([0] * cols)
  return matrix
