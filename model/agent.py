
# for kaggle-environments
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import sys

# finds all resources stored on the map and puts them into a list so we can search over them
def find_resources( game_state ):
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    
    for y in range( height ):
        for x in range( width ):
            cell = game_state.map.get_cell( x, y )
            if cell.has_resource():
                resource_tiles.append( cell )
    return resource_tiles

# finds the closest resources that we can mine given position on a map
def find_closest_resources( pos, player, resource_tiles ):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        # skip over resources that you can't mine due to not having researched them
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL \
        and not player.researched_coal(): 
            continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM\
        and not player.researched_uranium():
            continue
        
        dist = resource_tile.pos.distance_to( pos )
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    
    return closest_resource_tile

def agent( observation, configuration ):
    global game_state
    
    ### DO NOT EDIT - START ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize( observation["updates"] )
        game_state._update( observation["updates"][2:] )
        game_state.id = observation.player
    else:
        game_state._update( observation["updates"] )
        
    actions = []
    ### DO NOT EDIT - END ###
    
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    
    # add debug statements
    if game_state.turn == 0:
        print( "Agent is running!", file=sys.stderr )
        
    resource_tiles = find_resources( game_state )
    
    for unit in player.units:
        if unit.is_worker() and unit.can_act(): # worker(able to mine resources?), can_act(activated?)
            if unit.get_cargo_space_left() > 0: # space(storage capacity)
                closest_resource_tile = find_closest_resources( unit.pos, player, resource_tiles )
                if closest_resource_tile is not None:
                    action = unit.move( unit.pos.direction_to( closest_resource_tile.pos ) )
                    actions.append( action )
                    
            else:
                closest_city_tile = find_closest_city_tile( unit.pos, player )
                if closest_city_tile is not None:
                    action = unit.move( unit.pos.direction_to( closest_city_tile.pos ) )
                    actions.append( action )
                
    return actions
