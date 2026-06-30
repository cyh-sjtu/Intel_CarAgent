# CarAgent AsyncAgent Web Demo

Local web UI for sending messages to `AsyncAgent` and inspecting the current plan.

## Run

```bash
python -m caragent_agent.scripts.demo_ui.async_agent_web_demo \
  --dataset /home/car/caragent_ws/keyframes/session_20260524_005910/selected \
  --background-workers 2
```

Then open:

```text
http://127.0.0.1:8123
```

The demo uses a dry-run controller by default, so the UI can be exercised without Nav2. Use the ROS2 node for real `/navigate_to_pose` dispatch.
