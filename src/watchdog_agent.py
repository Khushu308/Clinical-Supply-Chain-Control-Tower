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

# # --- 1. TOOL DEFINITION ---
# def run_sql_query(query: str):
#     """
#     Connects to PostgreSQL, executes the query, and returns the results as a list of dictionaries.
#     """
#     conn = None
#     results = []
    
#     print(f"\n--- EXECUTING SQL QUERY ---\n{query}\n---------------------------\n")
    
#     try:
#         # Establish connection
#         conn = psycopg2.connect(**DB_CONFIG)
#         # Use DictCursor to return results as dictionaries (easier for JSON formatting)
#         with conn.cursor(cursor_factory=extras.DictCursor) as cur:
#             cur.execute(query)
#             # Fetch all rows
#             results = cur.fetchall()
            
#     except psycopg2.Error as e:
#         # Step 3: Edge Case Handling - Invalid SQL Query
#         # If an error occurs, the agent needs to 'self-heal' or report the failure.
#         print(f"!!! SQL EXECUTION ERROR: {e}")
#         # In a self-healing loop, this error would be passed back to the LLM
#         # to generate a corrected query, as described in Part 3 of the original assignment.
#         return None 
        
#     finally:
#         if conn:
#             conn.close()
            
#     return [dict(row) for row in results] # Convert DictRow objects to standard dictionaries

# # --- 2. AGENT LOGIC (The Orchestrator) ---
# def supply_watchdog_agent():
#     """
#     Orchestrates the daily Supply Watchdog autonomous workflow.
#     """
#     # --- A. Dynamic Context Generation (Runtime Parameters) ---
#     today = datetime.date.today() 
    
#     # 90-day Expiry Thresholds (Current Best Practice)
#     date_critical = today + datetime.timedelta(days=30)
#     date_high = today + datetime.timedelta(days=60)
#     date_warning = today + datetime.timedelta(days=90)
#     weeks_threshold = 8 # Shortfall threshold

#     # Determine dynamic demand columns based on current month (e.g., last 3 months)
#     current_month_index = today.month 
#     months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
#     # To get the last 3 *completed* months, we use indices [current_month_index - 4] to [current_month_index - 1]
#     # For December (index 12), this uses Sep, Oct, Nov. For January (index 1), this wraps to Oct, Nov, Dec of the previous year.
#     # The logic below simplifies to the last 3 in the list for demonstration, assuming month data is up-to-date.
#     if current_month_index > 3:
#          demand_cols = months[current_month_index-4:current_month_index-1]
#     else:
#          # Simplified wrap for Jan/Feb/Mar runs. In a real system, you'd pull from the previous 'Year' column too.
#          demand_cols = ['Oct', 'Nov', 'Dec'] 
         
#     # --- B. Execute Tasks ---
#     print(f"\n--- AGENT START: Daily Watchdog Run ({today}) ---")
    
#     # Task 1: Expiry Alert
#     expiry_sql = generate_expiry_sql(today, date_critical, date_high, date_warning)
#     results_expiry = run_sql_query(expiry_sql)

#     # Task 2: Shortfall Prediction
#     shortfall_sql = generate_shortfall_sql(demand_cols, weeks_threshold)
#     results_shortfall = run_sql_query(shortfall_sql)
    
#     # --- C. Final Output Generation ---
#     final_json = format_alert_json(today, results_expiry, results_shortfall)
    
#     print("\n========================================================")
#     print("--- FINAL WATCHDOG JSON PAYLOAD (for Email/API) ---")
#     print(final_json)
#     print("========================================================\n")


# # --- 3. DYNAMIC SQL GENERATORS (The Agent's LLM Logic) ---

# def generate_expiry_sql(today, critical, high, warning):
#     """Generates the Expiry Alert query using dynamic date parameters."""
#     return f"""
# -- WORKFLOW A.1: EXPIRY ALERT (Generated on {today.isoformat()})
# SELECT
#     a.trial_alias,
#     a.material_component_batch AS batch_id,
#     a.material_description,
#     b."Expiration date_shelf life"::date AS expiry_date,
#     SUM(a.order_quantity::INTEGER) AS quantity_at_risk,
#     CASE
#         WHEN b."Expiration date_shelf life"::date <= '{critical.isoformat()}' THEN 'Critical'
#         WHEN b."Expiration date_shelf life"::date <= '{high.isoformat()}' THEN 'High'
#         ELSE 'Warning'
#     END AS risk_level
# FROM allocated_materials_to_orders a
# JOIN batch_master b 
#     ON a.material_component_batch = b."Batch number"
# WHERE 
#     b."Expiration date_shelf life"::date <= '{warning.isoformat()}' -- <= 90 days total window
#     AND a.order_status NOT IN ('Shipped', 'Cancelled', 'Completed') -- Check active reservations
# GROUP BY 1, 2, 3, 4, 5, risk_level
# ORDER BY expiry_date ASC;
# """

# def generate_shortfall_sql(demand_cols, weeks_threshold):
#     """Generates the Shortfall Prediction query using dynamic demand columns."""
    
#     # E.g., (COALESCE("Sep",0) + COALESCE("Oct",0) + COALESCE("Nov",0))
#     demand_calc_str = ' + '.join([f"COALESCE(NULLIF(\"{col}\", '')::numeric, 0)" 
#     for col in demand_cols])
    
#     return f"""
# -- WORKFLOW A.2: SHORTFALL PREDICTION (Using demand columns: {demand_cols})
# WITH Trial_Map AS (
#     -- Maps Trial Name (Inventory) to Trial Alias (Enrollment)
#     SELECT DISTINCT trial_alias, trial_alias_description AS trial_name 
#     FROM allocated_materials_to_orders
# ),
# Supply AS (
#     -- Calculate current available stock
#     SELECT "Trial Name", "Location", SUM("Received Packages"::Integer - "Shipped Packages"::Integer) AS stock
#     FROM available_inventory_report
#     GROUP BY 1, 2
# ),
# Demand_Calc AS (
#     -- Calculate dynamic average monthly burn rate (last {len(demand_cols)} months)
#     SELECT 
#         "Trial Alias", 
#         "Country",
#         ({demand_calc_str}) / {len(demand_cols)}.0 AS avg_monthly_demand
#     FROM enrollment_rate_report
# )

# -- Final Join and Shortfall Calculation
# SELECT 
#     d."Trial Alias", 
#     s."Location" AS country,
#     s.stock AS available_stock, 
#     ROUND(d.avg_monthly_demand, 2) AS monthly_demand,
#     ROUND((s.stock / NULLIF(d.avg_monthly_demand, 0)) * 4, 1) AS weeks_coverage
# FROM Supply s
# JOIN Trial_Map m ON s."Trial Name" = m.trial_name
# JOIN Demand_Calc d 
#     ON m.trial_alias = d."Trial Alias" AND s."Location" = d."Country"
# WHERE 
#     d.avg_monthly_demand > 0 -- Exclude trials with zero demand (infinite coverage)
#     AND ((s.stock / NULLIF(d.avg_monthly_demand, 0)) * 4) < {weeks_threshold} -- Dynamic threshold check
# ORDER BY weeks_coverage ASC;
# """

# def format_alert_json(today, expiry_results, shortfall_results):
#     """Formats the raw query results into the structured JSON payload."""
    
#     import decimal

#     def convert_decimal(obj):
#         if isinstance(obj, list):
#             return [convert_decimal(i) for i in obj]
#         elif isinstance(obj, dict):
#             return {k: convert_decimal(v) for k, v in obj.items()}
#         elif isinstance(obj, decimal.Decimal):
#             return float(obj)
#         else:
#             return obj

#     expiry_results_clean = convert_decimal(expiry_results) if expiry_results is not None else []
#     shortfall_results_clean = convert_decimal(shortfall_results) if shortfall_results is not None else []

#     critical_expiry_count = sum(1 for r in expiry_results_clean if r.get('risk_level') == 'Critical') if expiry_results_clean else 0
#     shortfall_count = len(shortfall_results_clean) if shortfall_results_clean else 0

#     return json.dumps({
#         "alert_id": f"WATCHDOG_ALERT_{today.strftime('%Y%m%d')}",
#         "alert_timestamp": datetime.datetime.now().isoformat(),
#         "summary": {
#             "critical_expiry_risks": critical_expiry_count,
#             "shortfall_risks": shortfall_count,
#             "total_risks_found": critical_expiry_count + shortfall_count
#         },
#         "expiry_alerts": expiry_results_clean,
#         "shortfall_predictions": shortfall_results_clean
#     }, indent=2)

# # --- 4. EXECUTE THE AGENT ---
# supply_watchdog_agent() # Uncomment this line to run the code

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