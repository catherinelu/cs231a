DELETION_COST = 1
INSERTION_COST = 1
CHANGE_COST = 1

class Roster:
    def __init__(self, filename):
        self.roster = []
        self.roster.append('Adam Holmstead')  # TODO: Remove
        f = open(filename)
        for line in f:
            self.roster.append(line.strip())
        f.close()


    def get_closest_name(self, predicted_name):
        shortest_edit_distance = float('Inf')
        roster_name = None

        for i, name in enumerate(self.roster):
            print name
            name = name.replace(' ', '')
            cur_edit_distance = self._recursive_shortest_edit_distance(predicted_name,
                len(predicted_name), name, len(name), 0, 0, 0, shortest_edit_distance)
            cur_edit_distance /= float(len(name))  # Don't penalize long names
            if cur_edit_distance < shortest_edit_distance:
                shortest_edit_distance = cur_edit_distance
                roster_name = self.roster[i]

        print shortest_edit_distance, roster_name
        return roster_name

    
    def _recursive_shortest_edit_distance(self, predicted_name, predicted_name_length,
        possible_name, possible_name_length, i, j, cur_edit_distance, shortest_edit_distance):
        if i == predicted_name_length and j == possible_name_length:
            return cur_edit_distance
        elif i == predicted_name_length:
            return cur_edit_distance + (possible_name_length - j) * INSERTION_COST
        elif j == possible_name_length:
            return cur_edit_distance + (predicted_name_length - i) * DELETION_COST
        elif cur_edit_distance > shortest_edit_distance:  # Short-circuit
            return float('Inf')

        if predicted_name[i].lower() == possible_name[j].lower():
            return self._recursive_shortest_edit_distance(predicted_name, predicted_name_length,
                possible_name, possible_name_length, i + 1, j + 1, cur_edit_distance, shortest_edit_distance)
        else:
            delete_edit_distance = self._recursive_shortest_edit_distance(predicted_name,
                predicted_name_length, possible_name, possible_name_length, i + 1, j,
                cur_edit_distance + DELETION_COST, shortest_edit_distance)
            insertion_cost = self._recursive_shortest_edit_distance(predicted_name,
                predicted_name_length, possible_name, possible_name_length, i, j + 1,
                cur_edit_distance + INSERTION_COST, shortest_edit_distance)
            change_edit_distance = self._recursive_shortest_edit_distance(predicted_name,
                predicted_name_length, possible_name, possible_name_length, i + 1, j + 1,
                cur_edit_distance + CHANGE_COST, shortest_edit_distance)
            return min(delete_edit_distance, insertion_cost, change_edit_distance)