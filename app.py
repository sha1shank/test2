import gradio as gr
from warehouse_env.env import WarehouseFulfillmentEnv
from baseline import heuristic_agent


def run_demo(task_name: str):
    env = WarehouseFulfillmentEnv(task_name=task_name, seed=0)
    state = env.reset()
    steps = []
    while True:
        action = heuristic_agent(state.dict())
        state, reward, done, info = env.step(action)
        steps.append(f"{env.steps:02d} | {action} | reward={reward:.3f} | pos={state.robot_position} | inventory={state.inventory}")
        if done or env.steps >= env.max_steps:
            break

    score = round(env.run_episode(lambda observation: heuristic_agent(observation.dict()))["score"], 3)
    return (
        f"Task {task_name} complete. Score: {score}\n"
        f"Completed: {state.completed_items}\n"
        f"Steps: {env.steps}/{env.max_steps}\n"
        f"Inventory: {state.inventory}\n"
        f"Wrong drops: {env.dropped_wrong_items}",
        "\n".join(steps),
    )


demo = gr.Interface(
    fn=run_demo,
    inputs=gr.Radio(["easy", "medium", "hard"], label="Task Difficulty", value="easy"),
    outputs=[gr.Textbox(label="Baseline Summary", lines=6), gr.Textbox(label="Episode Trace", lines=18)],
    title="Warehouse Fulfillment OpenEnv Demo",
    description="Run a deterministic baseline agent through the Easy / Medium / Hard warehouse tasks.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=4444, share=True)
