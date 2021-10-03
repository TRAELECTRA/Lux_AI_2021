# Lux_AI_2021
https://www.kaggle.com/libbyu/trai-1/edit



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
