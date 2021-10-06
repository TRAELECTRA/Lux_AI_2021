# Lux_AI_2021
https://www.kaggle.com/c/lux-ai-2021/overview/description

# Bookmarks

* [Introduction](https://github.com/TRAELECTRA/Lux_AI_2021#Introduction)
* [Kits](https://github.com/TRAELECTRA/Lux_AI_2021#Kits)
* [Run](https://github.com/TRAELECTRA/Lux_AI_2021#Run)

# Introduction

<br>

### Game Resolution Order

1. CityTile actions along with increased cooldown
2. Unit actions along with increased cooldown
3. Roads are created
4. Resource collection
5. Resource drops on CityTiles
6. If night time, make Units consume resources and CityTiles consume fuel
7. Regrow wood tiles that are not depleted to 0
8. Cooldowns are handled / computed for each unit and CityTile

### City Tiles

* Worker
* Cart
* Research

### Units

Default: Cooldown < 1

* Type
  * Worker : 100 resources / unit
  * Cart : 2,000 resources / unit
* Action
  * Move : 5 방향 ( North, East, South, West, Center )
    * 상대편의 CityTile는 지나갈 수 없다.
    * CityTile에는 한 번에 복수의 unit이 있을 수 있지만, 다른 경우에는 불가능하다. (move 시도가 취소된다)
  * Carry : Mining 한 자원 + Transfer 한 자원
    * Worker / Cart가 CityTile에 자원을 가져가면, 가져간 자원을 연료로 바꿀 수 있다

### Workers' Actions

* Move
* Pillage(약탈) : Road Level을 낮춘다. ( 0.5 per unit )
* Transfer : Unit이 가지고 있는 한 종류의 자원을 양에 관계없이 인접한 Unit에 보낼 수 있다. ( 받는 사람의 용량을 초과한 양은 보낸 사람에게 되돌아온다. )
* Build CityTile : CityTile을 만들기 위해서는 Worker가 가진 자원이 100이어야 하고 (종류는 관계 없다), 해당 타일이 비어있어야 한다.

### Carts' Actions

* Move
* Transfer : Unit이 가지고 있는 한 종류의 자원을 양에 관계없이 인접한 Unit에 보낼 수 있다. ( 받는 사람의 용량을 초과한 양은 보낸 사람에게 되돌아온다. )

### Cooldown

* Unit이 Action을 수행한 이후 Cooldown이 실행된다.
* Unit이 Action을 수행한 이후 Cooldown 값이 증가한다.
  * CityTile : 10 per Action
  * Worker : 2 per Action
  * Cart : 3 per Action
* Unit과 CityTile들은 'Cooldown값 < 1'인 경우에만 Action을 실행할 수 있다.
* Cooldown Value
  * 도로가 건설된 후, pillage된 후, 각 턴마다 Unit들은 1만큼 Cooldown된다.
  * Unit이 턴이 끝나는 시점에 서있는 Road의 레벨만큼 Cooldown된다. ( CityTile )

### Roads

* Cart는 Unit이 빠르게 이동할 수 있게 만드는 Road를 만든다.
* Road Level은 0에서 시작하고, 6까지 증가할 수 있다.
* CityTile의 Road Level은 6이다.
* Worker는 pillage Action으로 0.5 per time 만큼 Road Level을 줄일 수 있다. ( 0이 되면, CityTile은 어둠에 빠진다 )
* 턴이 끝날때 Cart가 위치한 Road Level이 0.75만큼 증가한다.

### Day/Night Cycle

* 한 cycle은 40 turns (day: 30, night: 10) 으로 이루어진다. 
* 전체 게임은 총 9번의 cycle로 진행된다.
* 밤은 자원을 소모해서 생존을 위한 빛을 생산해야 한다. 
  * Citytile: 23 - 5 * (인접 타일 수) 
  * Units: 타일 위에 있지 않을때 cart 10, worker 4 
    * 가장 비효율적인 자원부터 소모 (wood > coal > uranium)하고, 해당 자원은 전부다 소모한다. 
  * Units는 base cooldown 2배 
  * 밤동안 자원이 부족한 unit은 게임에서 사라진다. 
  * 전체 도시의 자원이 부족한 경우 전체 citytile이 사라진다. 

# Kits

This folder contains all official kits provided by the Lux AI team for the Lux AI Challenge Season 1.

In each starter kit folder is a README and a folder called `simple` which gives you all the tools necessary to compete. Make sure to read the README document carefully. For debugging, you may log to standard error e.g. `console.error` or `print("hello", file=sys.stderr)`, this will be recorded and saved into a errorlogs folder for the match for each agent and will be recorded by the competition servers. You can also try out the [debug annotations](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Annotate) commands to draw onto the replay.

## API

This section details the general API each kit adheres to. It follows the conventions of Python but other kits are exactly the same, differing only in syntax and naming conventions. Any individual difference should be noted in the comments in the kits themselves.

Some methods you may notice are for generating an **action**, e.g. moving a [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit), transferring resources, building units. These action methods always return a string, which should be added to the `actions` array, of which there are examples of how to do so in the starter kits.

### game_state

The `game_state` object is provided to you and contains the complete information about the current state of the game at the current turn `game_state.turn`. Each agent/player in the game has an id, with your bot's id equal to `game_state.id` and the other team's id being `(game_state.id + 1) % 2`.

Additionally in the game state, are the following nested objects, `map` of type [GameMap](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#GameMap), and `players` which is a list with two [Player](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Player)objects indexed by the player's team id. The kits will show how to retrieve those objects. The rest of this section details the properties and methods of each type of object used in the kits.

### GameMap

The map is organized such that the top left corner of the map is at `(0, 0)` and the bottom right is at `(width - 1, height - 1)`. The map is always square.

Properties:

- `height: int` - the height of the map (along the y direction)
- `width: int` - the width of the map (along the x direction)
- `map: List[List[Cell]]` - A 2D array of [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) objects, defining the current state of the map. `map[y][x]` represents the cell at coordinates (x, y) with `map[0][0]` being the top left Cell.

Methods:

- `get_cell_by_pos(pos: Position) -> Cell` - returns the [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) at the given pos
- `get_cell(x: int, y: int) -> Cell` - returns the [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) at the given x, y coordinates

### Position

Properties:

- `x: int` - the x coordinate of the [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position)
- `y: int` - the y coordinate of the [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position)

Methods:

- `is_adjacent(pos: Position) -> bool` - returns true if this [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) is adjacent to `pos`. False otherwise
- `equals(pos: Position) -> bool` - returns true if this [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) is equal to the other `pos` object by checking x, y coordinates. False otherwise
- `translate(direction: DIRECTIONS, units: int) -> Position` - returns the [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) equal to going in a `direction` `units`number of times from this [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position)
- `distance_to(pos: Position) -> float` - returns the [Manhattan (rectilinear) distance](https://en.wikipedia.org/wiki/Taxicab_geometry) from this [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) to `pos`
- `direction_to(target_pos: Position) -> DIRECTIONS` - returns the direction that would move you closest to `target_pos` from this [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) if you took a single step. In particular, will return `DIRECTIONS.CENTER` if this [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) is equal to the `target_pos`. Note that this does not check for potential collisions with other units but serves as a basic pathfinding method

### Cell

Properties:

- `pos: Position`
- `resource: Resource` - contains details of a Resource at this [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell). This may be equal to `None` or `null` equivalents in other languages. You should always use the function `has_resource` to check if this [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) has a Resource or not
- `road: float` - the amount of Cooldown subtracted from a [Unit's](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) Cooldown whenever they perform an action on this tile. If there are roads, the more developed the road, the higher this Cooldown rate value is. Note that a [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) will always gain a base Cooldown amount whenever any action is performed.
- `citytile: CityTile` - the citytile that is on this [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell). Equal to `none` or `null` equivalents in other languages if there is no [CityTile](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#CityTile)here.

Methods:

- `has_resource() -> bool` - returns true if this [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) has a non-depleted Resource, false otherwise

### City

Properties:

- `cityid: str` - the id of this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City). Each [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) id in the game is unique and will never be reused by new cities
- `team: int` - the id of the team this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) belongs to.
- `fuel: float` - the fuel stored in this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City). This fuel is consumed by all CityTiles in this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) during each turn of night.
- `citytiles: list[CityTile]` - a list of [CityTile](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#CityTile) objects that form this one [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) collectively. A [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) is defined as all CityTiles that are connected via adjacent CityTiles.

Methods:

- `get_light_upkeep() -> float` - returns the light upkeep per turn of the [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City). Fuel in the [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) is subtracted by the light upkeep each turn of night.

### CityTile

Properties:

- `cityid: str` - the id of the [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) this [CityTile](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#CityTile) is a part of. Each [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) id in the game is unique and will never be reused by new cities
- `team: int` - the id of the team this [CityTile](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#CityTile) belongs to.
- `pos: Position` - the [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) of this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) on the map
- `cooldown: float` - the current Cooldown of this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City).

Methods:

- `can_act() -> bool` - whether this [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) can perform an action this turn, which is when the Cooldown is less than 1
- `research() -> str` - returns the research action
- `build_worker() -> str` - returns the build worker action. When applied and requirements are met, a worker will be built at the [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City).
- `build_cart() -> str` - returns the build cart action. When applied and requirements are met, a cart will be built at the [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City).

### Unit

Properties:

- `pos: Position` - the [Position](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Position) of this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) on the map
- `team: int` - the id of the team this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) belongs to.
- `id: str` - the id of this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit). This is unique and cannot be repeated by any other [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) or [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City)
- `cooldown: float` - the current Cooldown of this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit). Note that when this is less than 1, the [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) can perform an action
- `cargo.wood: int` - the amount of wood held by this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit)
- `cargo.coal: int` - the amount of coal held by this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit)
- `cargo.uranium: int` - the amount of uranium held by this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit)

Methods:

- `get_cargo_space_left(): int` - returns the amount of space left in the cargo of this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit). Note that any Resource takes up the same space, e.g. 70 wood takes up as much space as 70 uranium, but 70 uranium would produce much more fuel than wood when deposited at a [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City)
- `can_build(game_map: GameMap): bool` - returns true if the [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) can build a [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) on the tile it is on now. False otherwise. Checks that the tile does not have a Resource over it still and the [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) has a Cooldown of less than 1
- `can_act(): bool` - returns true if the [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) can perform an action. False otherwise. Essentially checks whether the Cooldown of the [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) is less than 1
- `move(dir): str` - returns the move action. When applied, [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) will move in the specified direction by one [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit), provided there are no other units in the way or opposition cities. ([Units](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) can stack on top of each other however when over a friendly [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City))
- `transfer(dest_id, resourceType, amount): str` - returns the transfer action. Will transfer from this [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) the selected Resource type by the desired amount to the [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) with id `dest_id` given that both units are adjacent at the start of the turn. (This means that a destination [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) can receive a transfer of resources by another [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) but also move away from that [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit))
- `build_city(): str` - returns the build [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) action. When applied, [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) will try to build a [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) right under itself provided it is an empty tile with no [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) or resources and the worker is carrying 100 units of resources. All resources are consumed if the city is succesfully built.
- `pillage(): str` - returns the pillage action. When applied, [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) will pillage the tile it is currently on top of and remove 0.5 of the road level.

### Player

This contains information on a particular player of a particular team.

Properties:

- `team: int` - the team id of this player
- `research_points: int` - the current total number of research points the player's team has
- `units: list[Unit]` - a list of every [Unit](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Unit) owned by this player's team.
- `cities: Dict[str, City]` - a dictionary / map mapping [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) id to each separate [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City) owned by this player's team. To get the individual CityTiles, you will need to access the `citytiles` property of the [City](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#City).

Methods:

- `researched_coal() - bool` - whether or not this player's team has researched coal and can mine coal.
- `researched_uranium() - bool` - whether or not this player's team has researched uranium and can mine uranium.

### Annotate

The annotation object lets you create annotation commands that show up on the visualizer when debug mode is turned on. Note that these commands are stripped by competition servers but are available to see when running matches locally.

Methods

- `circle(x: int, y: int) -> str` - returns the draw circle annotation action. Will draw a unit sized circle on the visualizer at the current turn centered at the [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) at the given x, y coordinates
- `x(x: int, y: int) -> str` - returns the draw X annotation action. Will draw a unit sized X on the visualizer at the current turn centered at the [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) at the given x, y coordinates
- `line(x1: int, y1: int, x2: int, y2: int) -> str` - returns the draw line annotation action. Will draw a line from the center of the [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) at (x1, y1) to the center of the [Cell](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits#Cell) at (x2, y2)
- `text(x: int, y: int, message: str, fontsize: int = 16) -> str:` - returns the draw text annotation action. Will write text on top of the tile at (x, y) with the particular message and fontsize
- `sidetext(message: str) -> str:` - returns the draw side text annotation action. Will write text that is displayed on that turn on the side of the visualizer

Note that all of these will be colored according to the team that created the annotation (blue or orange)

### GameConstants

This will contain constants on all game parameters like the max turns, the light upkeep of CityTiles etc.

If there are any crucial changes to the starter kits, typically only this object will change.

# Run

### Process

1. install npm version > 12.0
2. brew install lux-ai-2021
3. lux-ai-2021 [bot1] [bot2] --width width --height height
4. https://2021vis.lux-ai.org 에서 replay 파일 실행시키기
