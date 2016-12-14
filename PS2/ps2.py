# 6.0002 Problem Set 2
# Graph optimization
# Name: sco1
# Collaborators: N/A
# Time: 2:00

#
# Finding shortest paths through MIT buildings
#
import unittest
from graph import Digraph, Node, WeightedEdge

#
# Problem 2: Building up the Campus Map
#
# Problem 2a: Designing your graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the distances
# represented?
#
# Answer: The graph's nodes represent MIT's buildings. The edges represent valid
#         routes from one building to another whose distances, both outdoor and
#         indoor, are represented as edge weights.


# Problem 2b: Implementing load_map
def load_map(map_filename):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        map_filename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a Digraph representing the map
    """

    print("Loading map from file...")
    # Open map_filename for reading and parse each line to get the edge parameters
    # Assumes map_filename is properly formatted
    newdigraph = Digraph() # Initialize new digraph
    with open(map_filename, mode='r') as fID:
        for tline in fID:
            # Split on space
            edge_params = tline.split(' ')

            # edge_params will have the following format:
            #     edge_params[0]: Source Node, string
            #     edge_params[1]: Destination Node, string
            #     edge_params[2]: Total Distance, int
            #     edge_params[3]: Distance Outdoors, int
            for idx, param in enumerate(edge_params):
                if idx < 2:
                    # Node names, staying as strings. Strip newline
                    edge_params[idx] = edge_params[idx].replace('/n', '')
                else:
                    # Distances, cast as integers. This strips the newline
                    edge_params[idx] = int(edge_params[idx])
                    
            # Check to see if our source node is in the digraph we're generating,
            # add if it's not
            if not newdigraph.has_node(edge_params[0]):
                newdigraph.add_node(edge_params[0])

            # Check to see if our destination node is in the digraph we're
            # generating, add if it's not
            if not newdigraph.has_node(edge_params[1]):
                newdigraph.add_node(edge_params[1])

            newdigraph.add_edge(WeightedEdge(edge_params[0], edge_params[1], edge_params[2], edge_params[3]))

    return newdigraph


# Problem 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out
# test_digraph = load_map('.\\test_load_map.txt')
# print(test_digraph)

#
# Problem 3: Finding the Shorest Path using Optimized Search Method
#
# Problem 3a: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer: The objective functon for this problem is to minimize the travel 
#         distance between the two input buildings. Our constraints are the 
#         total travel distance along with the total distance traveled outdoors.

# Problem 3b: Implement get_best_path
def get_best_path(digraph, start, end, path, max_dist_outdoors, best_dist,
                  best_path):
    """
    Finds the shortest path between buildings subject to constraints.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        path: list composed of [[list of strings], int, int]
            Represents the current path of nodes being traversed. Contains
            a list of node names, total distance traveled, and total
            distance outdoors.
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path
        best_dist: int
            The smallest distance between the original start and end node
            for the initial problem that you are trying to solve
        best_path: list of strings
            The shortest path found so far between the original start
            and end node.

    Returns:
        A tuple with the shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k and the distance of that path.

        If there exists no path that satisfies and max_dist_outdoors 
        constraints, then return None.
    """
    for node in [start, end]:
        # Make sure our start and end nodes are in the digraph, raise a 
        # ValueError if they're not
        if not digraph.has_node(node):
            raise ValueError

    # Add our starting point to the list of traversed nodes
    path[0] = path[0] + [start]
    if start == end:
        # If start is the same as end then we've reached the destination
        return path
    else:
        for edge in digraph.get_edges_for_node(start):
            if edge.get_destination() not in path[0]:
                if best_path == None or len(path[0]) < len(best_path):
                    # Check the total distance traversed in order to stop 
                    # recursion if our total distance and/or total outdoor
                    # distances exceed their respective thresholds.
                    pathcopy = path.copy()  # Make a copy so the recursion restarts properly
                    if path[1] + edge.get_total_distance() <= best_dist and path[2] + edge.get_outdoor_distance() <= max_dist_outdoors:
                            pathcopy[1] += edge.get_total_distance()
                            pathcopy[2] += edge.get_outdoor_distance()
                            newPath = get_best_path(digraph, edge.get_destination(), end, pathcopy, max_dist_outdoors, best_dist, best_path)

                            if newPath != None:
                                best_path = newPath[0]
                                best_dist = newPath[1]

    return (best_path, best_dist)


# Problem 3c: Implement directed_dfs
def directed_dfs(digraph, start, end, max_total_dist, max_dist_outdoors):
    """
    Finds the shortest path from start to end using a directed depth-first
    search. The total distance traveled on the path must not
    exceed max_total_dist, and the distance spent outdoors on this path must
    not exceed max_dist_outdoors.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        max_total_dist: int
            Maximum total distance on a path
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then raises a ValueError.
    """
    # Call our recursive best path algorithm
    # Use max_total_dist as the initial best distance input, which forces the 
    # search algorithm to try and find paths that have an outdoor distance less
    # than max_total_dist
    dfs = get_best_path(digraph, start, end, [[], 0, 0], max_dist_outdoors, max_total_dist, None)

    if dfs[0] == None:
        # If get_best_path returns None for the given constraints, then no valid
        # path could be found. Raise a ValueError
        raise ValueError
    else:
        # Otherwise return the shortest path found
        return dfs[0]


# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

class Ps2Test(unittest.TestCase):
    LARGE_DIST = 99999

    def setUp(self):
        self.graph = load_map("mit_map.txt")

    def test_load_map_basic(self):
        self.assertTrue(isinstance(self.graph, Digraph))
        self.assertEqual(len(self.graph.nodes), 37)
        all_edges = []
        for _, edges in self.graph.edges.items():
            all_edges += edges  # edges must be dict of node -> list of edges
        all_edges = set(all_edges)
        self.assertEqual(len(all_edges), 129)

    def _print_path_description(self, start, end, total_dist, outdoor_dist):
        constraint = ""
        if outdoor_dist != Ps2Test.LARGE_DIST:
            constraint = "without walking more than {}m outdoors".format(
                outdoor_dist)
        if total_dist != Ps2Test.LARGE_DIST:
            if constraint:
                constraint += ' or {}m total'.format(total_dist)
            else:
                constraint = "without walking more than {}m total".format(
                    total_dist)

        print("------------------------")
        print("Shortest path from Building {} to {} {}".format(
            start, end, constraint))

    def _test_path(self,
                   expectedPath,
                   total_dist=LARGE_DIST,
                   outdoor_dist=LARGE_DIST):
        start, end = expectedPath[0], expectedPath[-1]
        self._print_path_description(start, end, total_dist, outdoor_dist)
        dfsPath = directed_dfs(self.graph, start, end, total_dist, outdoor_dist)
        print("Expected: ", expectedPath)
        print("DFS: ", dfsPath)
        self.assertEqual(expectedPath, dfsPath)

    def _test_impossible_path(self,
                              start,
                              end,
                              total_dist=LARGE_DIST,
                              outdoor_dist=LARGE_DIST):
        self._print_path_description(start, end, total_dist, outdoor_dist)
        with self.assertRaises(ValueError):
            directed_dfs(self.graph, start, end, total_dist, outdoor_dist)

    def test_path_one_step(self):
        self._test_path(expectedPath=['32', '56'])

    def test_path_no_outdoors(self):
        self._test_path(
            expectedPath=['32', '36', '26', '16', '56'], outdoor_dist=0)

    def test_path_multi_step(self):
        self._test_path(expectedPath=['2', '3', '7', '9'])

    def test_path_multi_step_no_outdoors(self):
        self._test_path(
            expectedPath=['2', '4', '10', '13', '9'], outdoor_dist=0)

    def test_path_multi_step2(self):
        self._test_path(expectedPath=['1', '4', '12', '32'])

    def test_path_multi_step_no_outdoors2(self):
        self._test_path(
            expectedPath=['1', '3', '10', '4', '12', '24', '34', '36', '32'],
            outdoor_dist=0)

    def test_impossible_path1(self):
        self._test_impossible_path('8', '50', outdoor_dist=0)

    def test_impossible_path2(self):
        self._test_impossible_path('10', '32', total_dist=100)


if __name__ == "__main__":
    unittest.main()
