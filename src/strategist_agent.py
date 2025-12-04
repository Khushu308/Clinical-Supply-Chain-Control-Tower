import datetime
import json
import psycopg2
import os # To retrieve the API key securely
from psycopg2 import extras
import google.generativeai as genai
import re

# Import the function to load environment variables
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load environment variables from the .env file
load_dotenv() 

# Retrieve configuration from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_DATABASE"), 
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT")
}
gemini_model = None
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error initializing Gemini model: {e}. Please ensure GEMINI_API_KEY is correct.")


# --- 2. AGENT TOOLS ---

def run_sql_query(query: str, db_config: dict):
    """
    TOOL: Connects to PostgreSQL, executes the query, and returns results as JSON string or an error message.
    """
    import decimal
    def convert_decimal(obj):
        if isinstance(obj, list):
            return [convert_decimal(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            return obj

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor(cursor_factory=extras.DictCursor) as cur:
            cur.execute(query)
            results = [dict(row) for row in cur.fetchall()]
            results = convert_decimal(results)
            return json.dumps(results), None
    except psycopg2.Error as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()

def generate_strategic_response(prompt_history: list, model_name: str = 'gemini-2.5-flash') -> str:
    """
    TOOL: Calls the Gemini LLM to generate the next action (SQL or FINAL_ANSWER).
    """
    if not gemini_model:
        raise ConnectionError("Gemini model not initialized.")
    try:
        # Concatenate prompt history as a single string
        prompt = "\n\n".join(prompt_history)
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0
            }
        )
        return response.text.strip()
    except Exception as e:
        raise Exception(f"LLM Generation Error: {e}")


# --- 3. DYNAMIC PROMPT & ORCHESTRATOR LOGIC ---

def get_strategist_system_prompt() -> str:
    """Defines the LLM's persona, logic, and required structured output format."""
    return """
You are the Scenario Strategist, an expert consultant analyzing the feasibility of shelf-life extension requests. Your output must ALWAYS be a single, structured block defining the NEXT action.

**REQUIRED OUTPUT FORMAT:**
If you need to query the database:
<ACTION>SQL_QUERY</ACTION>
<QUERY>SELECT ... FROM ...</QUERY>

You must perform each of the three checks (Technical, Logistical, Regulatory) exactly once. After each check, append the result (or error) to your reasoning. Do not repeat or rephrase queries for the same check. After all three checks, always return:
<ACTION>FINAL_ANSWER</ACTION>
<RESPONSE>Your final conversational answer, citing the result of each check. If a query fails, report the error and proceed to the next check. Never generate more than one query per check. Never exceed three queries before FINAL_ANSWER.

**CONSTRAINTS TO CHECK (CHECK ALL, REPORT ALL):**
1. **TECHNICAL:** Check batch_master for the "Expiration date_shelf life" and if an "Expiry Extension Date" already exists.
2. **LOGISTICAL:** Calculate the AVG lead time (actual_delivery_date - order_date) for the material's trial and compare it to the remaining shelf life.
3. **REGULATORY:** Verify the trial/material is active in the requested country by searching available_inventory_report.

If a SQL query fails, DO NOT try a different SQL for the same check. 
No EXTRACT needed in query.
Record the failure and proceed to the next check immediately.
Repeat checks is strictly forbidden.
Never generate more than 3 queries: 
1 Technical, 1 Logistical, 1 Regulatory.

## SCHEMA ABSTRACTION:
| Business Concept | Required Table | Key Columns |
| :--- | :--- | :--- |
| Batch Details | `batch_master` | "Batch number", "Expiration date_shelf life", "Expiry Extension Date" |
| Shipping Time | `distribution_order_report` | `trial_alias`, `order_date`, `actual_delivery_date` |
| Trial/Country Link | `available_inventory_report` | "Trial Name", "Location" |
| Material Link | `allocated_materials_to_orders` | `material_component_batch`, `trial_alias`, `trial_alias_description` |
Linkage: material_trial_link_view (L) | batch_number, trial_alias, country_name
START THE PROCESS by generating the query for the Technical Check.
"""

def parse_llm_action(llm_output: str) -> tuple:
    """Parses the LLM's structured output into an action and content."""
    action_match = re.search(r'<action>(.*?)</action>', llm_output, re.IGNORECASE | re.DOTALL)
    content_match = re.search(r'<(query|response)>(.*?)</(query|response)>', llm_output, re.IGNORECASE | re.DOTALL)

    action = action_match.group(1).strip() if action_match else 'ERROR'
    content = content_match.group(2).strip() if content_match else ''
    
    return action, content

def scenario_strategist_agent(user_query: str, db_config: dict, max_steps: int = 5):
    """
    Main orchestrator managing the conversational chain of reasoning with the LLM.
    """
    prompt_history = [
        get_strategist_system_prompt(),
        f"User Request: {user_query}"
    ]
    
    print(f"--- STRATEGIST AGENT INITIATED ---")
    print(f"User Query: {user_query}")
    
    for step in range(max_steps):
        print(f"\n--- STEP {step + 1}/{max_steps} ---")
        
        # 1. LLM Generates Next Action (Query or Answer)
        try:
            llm_output = generate_strategic_response(prompt_history)
        except Exception as e:
            return {"status": "Failure", "message": f"LLM generation failed: {e}"}

        # 2. Parse Action
        action, content = parse_llm_action(llm_output)
        
        # Add LLM's full output to history for context
        prompt_history = [
    get_strategist_system_prompt(),   # ← REINFORCE SYSTEM PROMPT EACH TURN
    f"User Request: {user_query}",
    f"Model Response: {llm_output}",
]

        
        if action == 'SQL_QUERY':
            print(f"Action: Generating and Executing SQL...")
            print(f"Query: {content}")
            
            # 3. Execute Tool Call
            db_results_json, db_error = run_sql_query(content, db_config)
            
            # 4. Feedback Loop
            if db_error:
                feedback = (
                    f"!!! SQL EXECUTION FAILED !!!\n"
                    f"The PostgreSQL error was: {db_error}\n"
                    f"ANALYZE THE ERROR AND GENERATE A CORRECTED SQL QUERY OR DECLARE FINAL_ANSWER IF BLOCKED."
                )
            else:
                feedback = (
                    f"Query Executed Successfully.\n"
                    f"DATABASE RESULT (JSON):\n{db_results_json}\n"
                    f"ANALYZE THE RESULT AND GENERATE THE NEXT SEQUENTIAL QUERY (LOGISTICAL or REGULATORY CHECK) OR DECLARE FINAL_ANSWER."
                )
            
            # Add tool output/feedback to history for the next turn
            prompt_history.append(f"<TOOL_RESPONSE>{feedback}</TOOL_RESPONSE>")
        
        elif action == 'FINAL_ANSWER':
            print("Action: Final Answer Ready.")
            return {"status": "Success", "response": content}
        
        else:
            print(f"Error: Unknown action '{action}' or parsing failed. Stopping.")
            return {"status": "Failure", "message": f"LLM output could not be parsed: {llm_output}"}

    return {"status": "Failure", "message": "Max execution steps reached without a final answer."}

# --- 5. EXECUTION EXAMPLE ---

# Example User Query
user_request = "Can we extend the expiry of Batch LOT-14364098 for the Taiwan trial? I need a clear YES or NO."

# Run the agent (requires valid API key and DB connection)
final_report = scenario_strategist_agent(user_request, DB_CONFIG)

print("\n---SCENARIO STRATEGIST REPORT ---")
print(json.dumps(final_report, indent=2))