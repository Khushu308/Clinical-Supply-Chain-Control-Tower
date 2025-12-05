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

import datetime
import json
import psycopg2
import os # To retrieve the API key securely
from psycopg2 import extras
import google.generativeai as genai

import re
# --- CONFIGURATION ---
gemini_model = None
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error initializing Gemini model: {e}. Please ensure GEMINI_API_KEY is correct.")

# --- 2. AGENT TOOLS ---

def run_sql_query(query: str, db_config: dict):
    """
    TOOL: Connects to PostgreSQL, executes the query, and returns results or an error message.
    """
    conn = None
    
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor(cursor_factory=extras.DictCursor) as cur:
            cur.execute(query)
            # Fetch column names for context if needed, but primarily fetch data
            results = cur.fetchall()
            return [dict(row) for row in results], None # Data, No Error
            
    except psycopg2.Error as e:
        # Capture the specific PostgreSQL error message for the LLM to analyze
        return None, str(e) # No Data, Error Message
        
    finally:
        if conn:
            conn.close()

def generate_sql_with_gemini(full_prompt: str, model_name: str = 'gemini-2.5-flash') -> str:
    """
    TOOL: Calls the Gemini LLM to generate the SQL query string.
    """
    if not gemini_model:
        raise ConnectionError("Gemini client not initialized due to missing API key or connection error.")
        
    try:
        response = gemini_model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.1
            }
        )
        sql_query = response.text.strip().replace('```sql', '').replace('```', '').strip()
        return sql_query
    except Exception as e:
        raise Exception(f"LLM Generation Error: {e}")

# --- 3. THE SELF-HEALING ORCHESTRATOR ---

def execute_with_retry(system_prompt: str, db_config: dict, max_retries: int = 3):
    """
    Implements the self-healing loop: LLM generates SQL -> Python executes -> 
    If SQL fails, Python feeds error back to LLM -> LLM corrects -> Retry.
    """
    current_prompt = system_prompt
    
    for attempt in range(max_retries):
        print(f"\n--- ATTEMPT {attempt + 1}/{max_retries}: Generating SQL ---")
        
        try:
            # 1. LLM Generates SQL
            sql_query = generate_sql_with_gemini(current_prompt)
            print(f"Generated Query:\n{sql_query}")
            
            # 2. Python Executes SQL
            results, error = run_sql_query(sql_query, db_config)
            
            if error is None:
                print("SQL Executed successfully.")
                return results # Success!
            
            # 3. SQL Failed - Prepare for Repair
            print(f"Execution failed with error: {error}")
            
            # 4. Agentic Repair Step: Update prompt with error context
            current_prompt += (
                f"\n\n!!! PREVIOUS ATTEMPT FAILED !!!\n"
                f"The generated SQL failed with the following PostgreSQL error:\n"
                f"{error}\n"
                f"The faulty query was:\n"
                f"{sql_query}\n"
                f"ANALYZE THE ERROR AND GENERATE A CORRECTED, COMPLETE SQL QUERY."
            )
        
        except Exception as e:
            print(f"A critical error occurred during generation or execution: {e}")
            break

    # If loop completes without success
    print("Failed to generate and execute a valid SQL query after all retries.")
    return None

# --- 4. THE MAIN AGENT WORKFLOW (Integrating the components) ---

def supply_watchdog_shortfall_workflow():
    """
    Prepares dynamic context and initiates the LLM-driven Shortfall Prediction.
    """
    if not gemini_model:
        return {"status": "error", "message": "Cannot run agent without Gemini model connection."}

    # --- A. Dynamic Context Generation (Python's Job) ---
    today = datetime.date.today() 
    weeks_threshold = 8 
    # Logic to dynamically determine demand columns (e.g., last 3 completed months)
    demand_cols = ['Sep', 'Oct', 'Nov'] # Example dynamic columns
    
    # --- B. System Prompt Construction (Python's Job) ---
    system_prompt = f"""
You are the Supply Watchdog, an autonomous, analytical AI agent. Your task is to perform daily inventory risk assessments and generate precise, executable PostgreSQL queries.
**CONTEXT:** Today is {today.isoformat()}. The lookback window for demand is the average of the last 3 months: **{demand_cols}**.
The alert threshold is stock running out in **{weeks_threshold} weeks** or less.
Your ONLY output must be the complete PostgreSQL SQL query. DO NOT include any explanatory text, markdown formatting (like ```sql`), or surrounding code.

## SCHEMA ABSTRACTION
You must join these tables (using the AS aliases provided) to solve the Shortfall Prediction:
| Business Concept | Required Table | Column Names (Use these exact names!) |
| :--- | :--- | :--- |
| **Supply** | `available_inventory_report` (S) | `"Trial Name"`, `"Location"`, `"Received Packages"`, `"Shipped Packages"` |
| **Demand** | `enrollment_rate_report` (D) | `"Trial Alias"`, `"Country"`, {', '.join(f'"{c}"' for c in demand_cols)}, `"Year"` |
| **Linkage** | `allocated_materials_to_orders` (M) | `trial_alias`, `trial_alias_description` (maps to S."Trial Name") |

## CALCULATION LOGIC:
1. Available Stock: SUM("Received Packages" - "Shipped Packages")
2. Monthly Demand: ({' + '.join(f'COALESCE("{c}",0)' for c in demand_cols)}) / {len(demand_cols)}.0
3. Weeks of Coverage: (Available Stock / NULLIF(Monthly Demand, 0)) * 4

## NOTE
- Safe Cast each column inside the math expressions to avoid null/empty string issues.
## FINAL INSTRUCTION:
Generate a single, complete PostgreSQL query that identifies all combinations of Trial/Country where 'Weeks of Coverage' is less than {weeks_threshold}. Use CTEs for clarity.
"""

    # --- C. Initiate Self-Healing Execution ---
    final_results = execute_with_retry(system_prompt, DB_CONFIG)

    # --- D. Final Output Formatting ---
    if final_results is not None:
        shortfall_alerts = [r for r in final_results if r.get('weeks_coverage', weeks_threshold + 1) < weeks_threshold]
        
        return {
            "report_date": today.isoformat(),
            "status": "Success",
            "shortfall_risks_found": len(shortfall_alerts),
            "shortfall_predictions": shortfall_alerts
        }
    else:
        return {
            "report_date": today.isoformat(),
            "status": "Failure",
            "message": "Agent failed to generate a valid SQL query against the database."
        }

# Execute the agent and print the final report
final_report = supply_watchdog_shortfall_workflow()
print("\n--- FINAL SHORTFALL REPORT ---")
print(json.dumps(final_report, indent=2))