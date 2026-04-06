import math
from typing import Any, Dict, List, Optional

from warehouse_env.env import WarehouseFulfillmentEnv
from warehouse_env.models import Action
from warehouse_env.tasks import TASKS


def _direction_toward(start: List[int], target: List[int]) -> str:
    if start[0] < target[0]:
        return "south"
    if start[0] > target[0]:
        return "north"
    if start[1] < target[1]:
        return "east"
    if start[1] > target[1]:
        return "west"
    return "wait"


def _closest_target(position: List[int], targets: List[List[int]]) -> Optional[List[int]]:
    if not targets:
        return None
    return min(targets, key=lambda t: abs(position[0] - t[0]) + abs(position[1] - t[1]))


def heuristic_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    position = state["robot_position"]
    inventory = state["inventory"]
    order_items = state["order_items"]
    completed_items = state["completed_items"]
    visible_items = state["visible_items"]
    max_steps = state["max_steps"]

    delivery_position = [0, 0]

    if inventory and position == delivery_position:
        return {"action_type": "drop", "direction": None}

    if any(item in order_items and item not in completed_items for item in inventory) and position != delivery_position:
        return {"action_type": "move", "direction": _direction_toward(position, delivery_position)}

    for item_id, item_pos in visible_items.items():
        if item_id in order_items and item_id not in completed_items and len(inventory) < 2 and position == item_pos:
            return {"action_type": "pick", "direction": None}

    available_targets = [pos for item_id, pos in visible_items.items() if item_id in order_items and item_id not in completed_items]
    if available_targets and len(inventory) < 2:
        target = _closest_target(position, available_targets)
        return {"action_type": "move", "direction": _direction_toward(position, target)}

    if inventory:
        return {"action_type": "move", "direction": _direction_toward(position, delivery_position)}

    if available_targets:
        target = _closest_target(position, available_targets)
        return {"action_type": "move", "direction": _direction_toward(position, target)}

    return {"action_type": "wait", "direction": None}


def run_baseline() -> None:
    print("Baseline inference results")
    print("==========================")
    for task_name in ["easy", "medium", "hard"]:
        env = WarehouseFulfillmentEnv(task_name=task_name, seed=0)
        result = env.run_episode(lambda observation: heuristic_agent(observation.dict()))
        print(f"Task: {task_name}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Steps: {result['steps']}/{TASKS[task_name]['max_steps']}")
        print(f"  Completed: {result['completed_items']}")
        print(f"  Total reward: {result['total_reward']:.3f}")
        print("")


if __name__ == "__main__":
    run_baseline()
