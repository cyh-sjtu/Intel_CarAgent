# Agent Audit Tools

These scripts are development-only checks for the async-agent workflow. They
were moved out of the `caragent_agent` Python package so they do not ship as
runtime modules.

Run them from the repository root with the agent package on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "ros2/caragent_ws/src/caragent_agent"
python tools/agent_audits/audit_prompt_contract_static.py
python tools/agent_audits/audit_planner_prompt_api.py
python tools/agent_audits/audit_background_preanalysis.py
```

On the Intel board:

```bash
cd /home/car/caragent_ws
PYTHONPATH=src/caragent_agent python3 tools/agent_audits/audit_prompt_contract_static.py
```

## Script Groups

- API audits: `audit_planner_prompt_api.py`.
- Offline workflow audits: `audit_background_preanalysis.py`,
  `audit_execute_failure_contract.py`, `audit_keyframe_prompt_contract.py`,
  `audit_navigation_arrival_gate.py`, `audit_decision_branch_repair.py`.
- Session/log audits: `audit_agent_session_replay.py`.
- Memory audits: `audit_run_memory.py`.
- Navigation simulation audits: `audit_nav2_simulation.py` requires a ROS2
  environment and checks workflow simulation behavior without dispatching real
  robot motion.
- Static prompt/runtime contract audit: `audit_prompt_contract_static.py`.

These tools may inspect legacy compatibility fields, but production prompts and
runtime model-facing guidance should use the current semantic-task contract:
`semantic_keyframe`, `semantic_object`, `outputs`, `inputs_from`, and
`submit_task_result`.
