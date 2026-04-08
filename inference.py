import os
import json
import time
from openai import OpenAI
from client import CleanxEnv
from models import CleanxAction

# Mandatory Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
# Requirements specify HF_TOKEN or OPENAI_API_KEY
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("Error: HF_TOKEN or OPENAI_API_KEY must be set.")
    exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """
You are CleanX AI Data Fixer. Your job is to clean dirty datasets step-by-step.
You must output a single JSON action object on each turn.

Allowed operations:
1. {"operation": "drop_row", "args": {"dropna": true}} -> Drops all rows with missing values
2. {"operation": "drop_row", "args": {"drop_duplicates": true}} -> Drops duplicate rows
3. {"operation": "rename_column", "args": {"old_name": "x", "new_name": "y"}} -> Renames a column
4. {"operation": "cast_type", "args": {"column": "x", "type": "datetime"}} -> Valid types: "datetime", "float", "bool", "int", "str"
5. {"operation": "submit", "args": {}} -> Submit when finished

DO NOT output anything other than standard JSON. DO NOT wrap JSON in code blocks.
"""

MAX_STEPS = 10

def build_user_prompt(step, observation):
    error_str = f"Last Error: {observation.last_action_error}\n" if observation.last_action_error else ""
    return f"""
Step {step}
{error_str}
Goal: {observation.goal}
Columns: {observation.columns}
Current Score: {observation.progress}

Dataset Preview (CSV):
{observation.dataset_preview[:1000]}

What is your next JSON action?
"""

def parse_model_action(response_text: str) -> CleanxAction:
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
             text = text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(text)
        return CleanxAction(operation=data.get("operation", "none"), args=data.get("args", {}))
    except Exception as e:
        print(f"DEBUG: Failed to parse: {e} | Raw: {response_text}")
        return CleanxAction(operation="none", args={})


async def run_task(env: CleanxEnv, task_id: int):
    # Mandatory START log
    print(f"[START] task_id={task_id}")
    
    result = await env.reset()
    observation = result.observation
    
    for step in range(1, MAX_STEPS + 1):
        prompt = build_user_prompt(step, observation)
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            response = completion.choices[0].message.content or ""
        except Exception as e:
            response = '{"operation": "none", "args": {}}'

        action = parse_model_action(response)
        result = await env.step(action)
        observation = result.observation

        # Mandatory STEP log
        action_json = json.dumps({"operation": action.operation, "args": action.args})
        obs_json = json.dumps({
            "columns": observation.columns,
            "progress": observation.progress,
            "error": observation.last_action_error
        })
        print(f"[STEP] step={step} action={action_json} observation={obs_json} reward={result.reward} done={result.done}")

        if result.done:
            break

    # Mandatory END log
    print(f"[END] task_id={task_id} score={observation.progress}")

import asyncio

async def main_async():
    # Environment usually runs on localhost:8000 during local validation
    env = CleanxEnv(base_url=os.environ.get("ENV_URL", "http://localhost:8000"))
    
    # Run 3 episodes to cover Easy, Medium, Hard (randomly selected by env)
    for i in range(1, 4):
        try:
            await run_task(env, i)
        except Exception as e:
            print(f"Error in task {i}: {e}")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
