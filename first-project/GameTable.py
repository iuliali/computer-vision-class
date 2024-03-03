

class GameTable:
    TABLE_POINTS_POS_quarter = {
        (0, 0, 5),
        (0, 3, 4),
        (1, 2, 3),
        (1, 5, 4),
        (2, 1, 3),
        (2, 4, 2),
        (3, 0, 4),
        (3, 3, 3),
        (3, 5, 2),
        (4, 2, 2),
        (4, 4, 1),
        (4, 6, 1),
        (5, 1, 4),
        (5, 3, 2),
        (5, 5, 1),
        (6, 4, 1),
    }
    MATRIX_POINTS_POS_quarter = []
    NUM_LINE_COLUMNS = 15
    A_LETTER_ASCII = 65
    BONUS_POINTS_MATCH = 3
    SCORE_TRAIL = [1, 2, 3, 4, 5, 6,
                   0, 2, 5, 3, 4, 6,
                   2, 2, 0, 3, 5, 4,
                   1, 6, 2, 4, 5, 5,
                   0, 6, 3, 4, 2, 0,
                   1, 5, 1, 3, 4, 4,
                   4, 5, 0, 6, 3, 5,
                   4, 1, 3, 2, 0, 0,
                   1, 1, 2, 3, 6, 3,
                   5, 2, 1, 0, 6, 6,
                   5, 2, 1, 2, 5, 0,
                   3, 3, 5, 0, 6, 1,
                   4, 0, 6, 3, 5, 1,
                   4, 2, 6, 2, 3, 1,
                   6, 5, 6, 2, 0, 4,
                   0, 1, 6, 4, 4, 1,
                   6, 6, 3, 0]

    def __init__(self, players=None):
        if players is None:
            players = ["player1", "player2"]
        self.game_table = [[None for _ in range(GameTable.NUM_LINE_COLUMNS)]
                           for _ in range(GameTable.NUM_LINE_COLUMNS)]
        GameTable.make_matrix()
        self.players = players
        self.score = {player: 0 for player in players}
        self.current_player = "player1"

    @staticmethod
    def line_to_actual_line(line):
        return line + 1

    @staticmethod
    def column_to_actual_column(column):
        return chr(column + GameTable.A_LETTER_ASCII)

    @staticmethod
    def make_matrix():
        GameTable.MATRIX_POINTS_POS_quarter = [[0 for _ in range(GameTable.NUM_LINE_COLUMNS // 2)]
                                               for _ in range(GameTable.NUM_LINE_COLUMNS // 2)]
        for i, j, points in GameTable.TABLE_POINTS_POS_quarter:
            GameTable.MATRIX_POINTS_POS_quarter[i][j] = points

    @staticmethod
    def position_to_points(position):
        x, y = position
        if (x == 7 and (y == 0 or y == 14)) or (y == 7 and (x == 0 or x == 14)):
            return 3
        if x == 7 or y == 7:
            return 0
        if x > 7:
            x = GameTable.NUM_LINE_COLUMNS - 1 - x  # simetricul fata de mijloc pe ox
        if y > 7:
            y = GameTable.NUM_LINE_COLUMNS - 1 - y  # simetricul fata de mijloc pe oy
        return GameTable.MATRIX_POINTS_POS_quarter[x][y]

    def get_number_on_score_trail(self, player):
        if self.score[player] <= 0 or self.score[player] > 99:
            return None
        return GameTable.SCORE_TRAIL[self.score[player] - 1]

    def score_trail_to_points(self, piece_added):
        score_this_turn = {player: 0 for player in self.players}
        for player in self.players:
            number_score_trail = self.get_number_on_score_trail(player)
            if number_score_trail is not None and number_score_trail in list(piece_added):
                score_this_turn[player] = GameTable.BONUS_POINTS_MATCH
        return score_this_turn

    def is_double(self, piece):
        return piece[0] == piece[1]

    def add_score_points_to_player(self, positions_piece_added, piece_added):
        # Returns points for player from current move
        points_from_position = 0
        for pos in positions_piece_added:
            points_from_position += GameTable.position_to_points(pos)

        if self.is_double(piece_added):
            points_from_position *= 2
        print("Points from placing piece ", points_from_position)

        # Take score from trail
        score_this_turn = self.score_trail_to_points(piece_added)
        print("Points from BONUS ", score_this_turn[self.current_player])

        # Add score for move for current player
        score_this_turn[self.current_player] += points_from_position
        print("Total POINTS ", score_this_turn[self.current_player])

        # Advance with both players by increasing score
        for p in self.players:
            self.score[p] += score_this_turn[p]
        print(f"SCOR: {self.score}")
        # Return score this turn for current player
        return score_this_turn[self.current_player]

    def get_output_current_move(self, added_positions, write=False, file_to_write='1_01.txt'):
        if len(added_positions) < 2:
            if write:
                with open(file_to_write, "w") as file:
                    file.write(f"{0}\n"
                               f"{0}\n"
                               f"{0}\n")
            return 0, 0, 0

        piece_added = []
        for pos in added_positions[:2]:
            piece_added.append(self.game_table[pos[0]][pos[1]])

        pos1, pos2 = added_positions[:2]
        score = self.add_score_points_to_player(positions_piece_added=added_positions,
                                                piece_added=tuple(piece_added))
        if write:
            with open(file_to_write, "w") as file:
                file.write(f"{GameTable.line_to_actual_line(pos1[0])}{GameTable.column_to_actual_column(pos1[1])} "
                           f"{self.game_table[pos1[0]][pos1[1]]}\n"
                           f"{GameTable.line_to_actual_line(pos2[0])}{GameTable.column_to_actual_column(pos2[1])} "
                           f"{self.game_table[pos2[0]][pos2[1]]}\n"
                           f"{score}")
        return (pos1, self.game_table[pos1[0]][pos1[1]]), (pos2, self.game_table[pos2[0]][pos2[1]]), score
