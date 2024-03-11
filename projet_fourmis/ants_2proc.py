"""
Module managing an ant colony in a labyrinth.
"""
from mpi4py import MPI
import time
import sys
import numpy as np
import maze
import pheromone
import direction as d
import pygame as pg

UNLOADED, LOADED = False, True

exploration_coefs = 0.


class Colony:
    """
    Represent an ant colony. Ants are not individualized for performance reasons!

    Inputs :
        nb_ants  : Number of ants in the anthill
        pos_init : Initial positions of ants (anthill position)
        max_life : Maximum life that ants can reach
    """
    def __init__(self, nb_ants, pos_init, max_life):
        # Each ant has is own unique random seed
        self.seeds = np.arange(1, nb_ants+1, dtype=np.int64)
        # State of each ant : loaded or unloaded
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        # Compute the maximal life amount for each ant :
        #   Updating the random seed :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Amount of life for each ant = 75% à 100% of maximal ants life
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life*(self.seeds/2147483647.))//4
        # Ages of ants : zero at beginning
        self.age = np.zeros(nb_ants, dtype=np.int64)
        # History of the path taken by each ant. The position at the ant's age represents its current position.
        self.historic_path = np.zeros((nb_ants, max_life+1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction in which the ant is currently facing (depends on the direction it came from).
        self.directions = d.DIR_NONE*np.ones(nb_ants, dtype=np.int8)
        # Chargement de l'image uniquement sur le processus principal (rank 0)
        if rank == 0:
            img = pg.image.load("ants.png").convert_alpha()
            self.sprites = np.array([pg.Surface.subsurface(img, i, 0, 8, 8) for i in range(0, 32, 8)])
        else:
            self.sprites = []

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Function that returns the ants carrying food to their nests.

        Inputs :
            loaded_ants: Indices of ants carrying food
            pos_nest: Position of the nest where ants should go
            food_counter: Current quantity of food in the nest

        Returns the new quantity of food
        """
        self.age[loaded_ants] -= 1

        in_nest_tmp = self.historic_path[loaded_ants, self.age[loaded_ants], :] == pos_nest
        if in_nest_tmp.any():
            in_nest_loc = np.nonzero(np.logical_and(in_nest_tmp[:, 0], in_nest_tmp[:, 1]))[0]
            if in_nest_loc.shape[0] > 0:
                in_nest = loaded_ants[in_nest_loc]
                self.is_loaded[in_nest] = UNLOADED
                self.age[in_nest] = 0
                food_counter += in_nest_loc.shape[0]
        return food_counter

    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        """
        Management of unloaded ants exploring the maze.

        Inputs:
            unloadedAnts: Indices of ants that are not loaded
            maze        : The maze in which ants move
            posFood     : Position of food in the maze
            posNest     : Position of the ants' nest in the maze
            pheromones  : The pheromone map (which also has ghost cells for
                          easier edge management)

        Outputs: None
        """
        # Update of the random seed (for manual pseudo-random) applied to all unloaded ants
        self.seeds[unloaded_ants] = np.mod(16807*self.seeds[unloaded_ants], 2147483647)

        # Calculating possible exits for each ant in the maze:
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0

        # Reading neighboring pheromones:
        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]]*has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]]*has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]]*has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]]*has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        # Calculating choices for all ants not carrying food (for others, we calculate but it doesn't matter)
        choices = self.seeds[:] / 2147483647.

        # Ants explore the maze by choice or if no pheromone can guide them:
        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                # Calculating indices of ants whose last move was not valid:
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # Choosing a random direction:
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                # Valid move if we didn't stay in place due to a wall
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0], new_pos[:, 1] != old_pos[:, 1])
                # and if we're not in the opposite direction of the previous move (and if there are other exits)
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3-self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                # Calculating indices of ants whose move we just validated:
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                # For these ants, we update their positions and directions
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] -= max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] += max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        # Aging one unit for the age of ants not carrying food
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Killing ants at the end of their life:
        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        # For ants reaching food, we update their states:
        ants_at_food_loc = np.nonzero(np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
                                                     self.historic_path[unloaded_ants, self.age[unloaded_ants], 1] == pos_food[1]))[0]
        if ants_at_food_loc.shape[0] > 0:
            ants_at_food = unloaded_ants[ants_at_food_loc]
            self.is_loaded[ants_at_food] = True

    def advance(self, the_maze, pos_food, pos_nest, pheromones, food_counter=0):
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze.WEST) > 0
        # Marking pheromones:
        [pheromones.mark(self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]]) for i in range(self.directions.shape[0])]
        return food_counter

    def display(self, screen):
        [screen.blit(self.sprites[self.directions[i]], (8*self.historic_path[i, self.age[i], 1], 8*self.historic_path[i, self.age[i], 0])) for i in range(self.directions.shape[0])]

# # Fonction pour diviser les fourmis entre les processus MPI
# def distribute_ants(total_ants, num_processes, process_rank):
#     if process_rank == 0:
#         return 0, 0  # Le processeur 0 n'a pas besoin de fourmis
#     ants_per_process = total_ants // (num_processes - 1)
#     remainder = total_ants % (num_processes - 1)
#     start_ant = (process_rank - 1) * ants_per_process + min(process_rank - 1, remainder)
#     end_ant = start_ant + ants_per_process + (1 if process_rank - 1 < remainder else 0)
#     return start_ant, end_ant




if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    print(f"size = {size}\n")
    rank = comm.Get_rank()

    pg.init()

    # Initialisation des variables globales
    size_laby = 25, 25
    if len(sys.argv) > 2:
        size_laby = int(sys.argv[1]),int(sys.argv[2])
    
    max_life = 500
    if len(sys.argv) > 3:
        max_life = int(sys.argv[3])
    
    alpha = 0.9
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
    
    beta = 0.99
    if len(sys.argv) > 5:
        beta = float(sys.argv[5])

    resolution = size_laby[1]*8, size_laby[0]*8
    nb_ants = size_laby[0]*size_laby[1] // 4
    pos_food = size_laby[0]-1, size_laby[1]-1
    pos_nest = 0, 0
    food_counter = 0

    if rank==0:
        screen = pg.display.set_mode(resolution)

    # Initialisation de la colonie pour chaque processus
    # start_ant, end_ant = distribute_ants(nb_ants, size, rank)
    # local_ants_count = end_ant - start_ant
    # local_colony = Colony(local_ants_count, pos_nest, max_life)
    local_colony = Colony(nb_ants, pos_nest, max_life)
    # colony_data = {
    #         'seeds': local_colony.seeds,
    #         'is_loaded': local_colony.is_loaded,
    #         'max_life': local_colony.max_life,
    #         'age': local_colony.age,
    #         'historic_path': local_colony.historic_path,
    #         'directions': local_colony.directions,
    #         'sprites': local_colony.sprites
    #         }
    # print(rank, colony_data)
    # print(rank,colony_data['max_life']
    # colony_data['seeds']
    # colony_data['is_loaded']
    # colony_data['age']
    # colony_data['historic_path']
    # colony_data['directions']

    # Initialisation du labyrinthe pour chaque processus
    a_maze = maze.Maze(size_laby, 12345, rank)
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)

    if rank==0:
        global_colony = local_colony
        iteration = 0

    snapshop_taken = False
    while True:
        if rank==0:
            # print(f"iteration = {iteration}")
            # iteration+=1
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    comm.Abort(0)

        # deb = time.time()
        
        # Calculs de la colonie locale
        # Sur le processeur 1, après chaque itération de calculs :
        if rank != 0:
            food_counter = local_colony.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
            pherom.do_evaporation(pos_food)
            # Avant d'envoyer la local_colony
            colony_data = {
            'seeds': local_colony.seeds,
            'is_loaded': local_colony.is_loaded,
            'max_life': local_colony.max_life,
            'age': local_colony.age,
            'historic_path': local_colony.historic_path,
            'directions': local_colony.directions,
            }
            # print(f"proc{rank} : j'envoie mes données au proc 0")

            comm.send(colony_data, dest=0, tag=2)
            
            # print(f"proc{rank} : mes données ont bien été reçues.")

            comm.send(pherom.pheromon,dest=0, tag=4)
            
            # print(f"proc{rank} : j'envoie le food_counter au proc 0")

            comm.send(food_counter, dest=0, tag=1)

            # print(f"proc{rank} : mon food counter a bien été reçu.")


        # Sur le processeur 0 :
        if rank == 0:
            # Recevoir les données mises à jour de Colony depuis les autres processeurs
            for i in range(1, size):
                # print(f"proc {rank} : je demande à recevoir les données du proc {i}")
                colony_data = comm.recv(source=i, tag=2)
                # print(f"proc {rank} : je les ai bien reçues")
                # Mettre à jour l'instance de Colony du processeur 0 avec les données reçues
                global_colony.max_life = colony_data['max_life']
                global_colony.seeds = colony_data['seeds']
                global_colony.is_loaded = colony_data['is_loaded']
                global_colony.age = colony_data['age']
                global_colony.historic_path = colony_data['historic_path']
                global_colony.directions = colony_data['directions']

                pherom_data = comm.recv(source=i, tag=4)
                pherom.pheromon=pherom_data

                food_counter += comm.recv(source=i, tag=1)
                print(food_counter, end='\r')


            # Après avoir synchronisé toutes les instances de Colony, le processeur 0 peut afficher les fourmis.
            a_maze_img = a_maze.display()
            pherom.display(screen)
            screen.blit(a_maze_img, (0, 0))
            global_colony.display(screen)
            pg.display.update()

            # Sauvegarde de l'image une fois que la nourriture est collectée pour la première fois
            if food_counter == 1 and not snapshop_taken:
                pg.image.save(screen, "MyFirstFood.png")
                snapshop_taken = True

            # Affichage des informations
            # print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", end='\r')
        # time.sleep(5)
        

    # Terminaison de Pygame
    pg.quit()