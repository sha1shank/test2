# Warehouse Order Fulfillment OpenEnv

This repository defines a real-world OpenEnv environment for autonomous warehouse order fulfillment.
The agent must move a robot across a warehouse grid, pick items in a targeted order, and deliver them to a shipping pallet.

## Environment

- `openenv.yaml` describes the environment entrypoint and expected action/observation structure.
- `warehouse_env/` contains the typed models, environment implementation, task definitions, and grading logic.

## Task curriculum

- `easy`: 2-item order, same zone, low step budget.
- `medium`: 3-item order across two zones, moderate step budget.
- `hard`: 5-item order with decoy items and longer planning horizon.

## Action space

An action is a JSON object with:
- `action_type`: one of `move`, `pick`, `drop`, `wait`
- `direction`: required for `move`, one of `north`, `east`, `south`, `west`

Example:
```json
{"action_type": "move", "direction": "north"}
```

## Observation space

Observations are typed objects with:
- `robot_position`: `[row, col]`
- `inventory`: list of items currently carried
- `order_items`: full order item list
- `completed_items`: delivered order items
- `visible_items`: map from item IDs to shelf coordinates
- `time_step`: current step index
- `max_steps`: task budget
- `task_name`: current difficulty
- `remaining_distance`: Manhattan distance to the next objective

## Reward

The reward is continuous in `[0.0, 1.0]` with partial progress signals:
- correct items picked and delivered contribute strongly
- items in inventory contribute moderately
- proximity to next required item or delivery point is rewarded
- wrong item deliveries reduce reward
- full successful fulfillment reaches `1.0`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the baseline

```bash
python baseline.py
```

## Run the Hugging Face Spaces demo locally

```bash
python app.py
```

Then open the displayed `gradio` URL.

## Deployment

This repository includes a working `Dockerfile` for container deployment.
A deployed Hugging Face Space can be built from this repo with the default `app.py` interface.
