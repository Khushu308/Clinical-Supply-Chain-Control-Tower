import os
import json
import datetime
import psycopg2
from psycopg2 import extras
from typing import TypedDict, Annotated, List, Union, Literal
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. CONFIGURATION ---
load_dotenv()

# Database Config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_DATABASE"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT")
}

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    """The state of the watchdog agent loop."""
    messages: List[BaseMessage]
    retry_count: int
    max_retries: int
    # Consolidated results for final report
    final_report: dict
    # Current operation context
    current_operation: Literal["EXPIRY_ALERT", "SHORTFALL_PREDICTION"]
    # Temporary fields for query execution
    query_result: Union[List[dict], None]
    error: Union[str, None]

# --- 3. HELPER FUNCTION ---
def run_sql_query_safe(query: str) -> tuple[List[dict] | None, str | None]:
    """Helper to execute SQL queries safely, returns (results, error)."""
    def convert_decimal(obj):
        # Convert Decimals to float for JSON serialization
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return obj

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=extras.DictCursor) as cur:
            cur.execute(query)
            if cur.description:
                results = [dict(row) for row in cur.fetchall()]
                # Ensure conversion
                clean_results = json.loads(json.dumps(results, default=convert_decimal))
                return clean_results, None
            else:
                conn.commit()
                return [], None
    except Exception as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()

# --- 4. GRAPH NODES ---


import time

def safe_llm_invoke(llm, messages):
    retry_count = 0
    while retry_count < 5:
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                wait = 30  # wait seconds
                print(f"Quota exceeded. Waiting {wait}s before retrying...")
                time.sleep(wait)
                retry_count += 1
            else:
                raise
    raise RuntimeError("Failed to invoke LLM after retries due to quota limits.")


# 4.1. Node: Generate SQL Query
def generate_query_node(state: AgentState):
    """LLM node to generate the SQL query based on the current context."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # response = llm.invoke(state["messages"])
    response = safe_llm_invoke(llm, state["messages"])
    # Convert the LLM output to an AIMessage
    ai_msg = AIMessage(content=response.text if hasattr(response, "text") else str(response))
    # Append to existing messages
    return {"messages": state["messages"] + [ai_msg]}


# 4.2. Node: Execute SQL Query
def execute_query_node(state: AgentState):
    """Executes the generated SQL query."""
    last_message = state["messages"][-1]
    sql_query = last_message.content.replace('```sql', '').replace('```', '').strip()
    
    results, error_msg = run_sql_query_safe(sql_query)
    
    if error_msg:
        # Failure: capture error and increment retry count
        return {
            "error": error_msg,
            "retry_count": state["retry_count"] + 1,
            "query_result": None
        }
    else:
        # Success: store results
        return {
            "query_result": results, 
            "error": None
        }

# 4.3. Node: Prepare Retry Message
def prepare_retry_message(state: AgentState):
    """Creates the feedback message for the LLM upon failure."""
    error_msg = state["error"]
    last_query = state["messages"][-1].content
    
    feedback = (
        f"!!! PREVIOUS ATTEMPT FAILED (Attempt {state['retry_count']})!!!\n"
        f"The generated SQL failed with the following PostgreSQL error:\n"
        f"{error_msg}\n"
        f"The faulty query was:\n"
        f"```sql\n{last_query}\n```\n"
        f"ANALYZE THE ERROR and the full SYSTEM PROMPT. **GENERATE A CORRECTED, COMPLETE SQL QUERY.**"
    )
    # Append retry message to history
    return {"messages": [HumanMessage(content=feedback)]}

# 4.4. Node: Process Results & Prepare Next Step (Core Logic)
def process_results_and_advance(state: AgentState):
    """
    Saves the results of the current operation and transitions to the next operation or END.
    """
    operation = state["current_operation"]
    results = state["query_result"]
    report = state["final_report"]
    if operation == "EXPIRY_ALERT":
        # ...
        new_system_prompt = generate_system_prompt("EXPIRY_ALERT")
        
        return {
            # ...
            "messages": [
                SystemMessage(content=new_system_prompt),
                HumanMessage(content="The first stage is complete. Generate the SQL query for the SHORTFALL PREDICTION now, strictly following the new System Prompt and Schema.")
            ], 
        }
        
    elif operation == "SHORTFALL_PREDICTION":
        # Save Shortfall Prediction results
        report["shortfall_predictions"] = results or []
        return {
            "final_report": report,
        }

# --- 5. CONDITIONAL EDGES ---

def check_execution_status(state: AgentState):
    """Determines if the current query execution succeeded, needs retry, or failed permanently."""
    if state["query_result"] is not None:
        # Query succeeded, move to process results and advance
        return "SUCCESS"
    
    if state["retry_count"] >= state["max_retries"]:
        # Failed permanently, move to process results (and save error)
        return "MAX_RETRIES_REACHED"
    
    # Failed, prepare for retry
    return "RETRY"

def check_workflow_stage(state: AgentState):
    """Determines the next major workflow stage."""
    if state["current_operation"] == "SHORTFALL_PREDICTION":
        # Last stage complete, move to END
        return "END"
    # Otherwise, continue to the next step which is END
    return "ADVANCE"


# --- 6. SYSTEM PROMPT GENERATOR (Combines Expiry & Shortfall) ---

def generate_system_prompt(stage: Literal["EXPIRY_ALERT", "SHORTFALL_PREDICTION"]) -> str:
    """Generates the context-specific prompt for the LLM."""
    today = datetime.date.today()
    
    if stage == "EXPIRY_ALERT":
        # Logic from the 'Query Expiry Risks' node
        expiry_threshold_days = 90
        return f"""
You are the Supply Watchdog, responsible for generating precise PostgreSQL queries.
**CONTEXT:** Today is {today.isoformat()}.
Your current task is **EXPIRY ALERT**: Identify all inventory items that will expire within the next {expiry_threshold_days} days.
Your ONLY output must be the complete PostgreSQL SQL query. DO NOT include any explanatory text, markdown formatting (like ```sql), or surrounding code.

## SCHEMA ABSTRACTION
Table: `available_inventory_report` (S)
Key Columns: `"Trial Name"`, `"Material Number"`, `"Batch/Lot Number"`, `"Expiry Date"`, `"Received Packages"`, `"Shipped Packages"`.

## CALCULATION LOGIC:
1. Days Until Expiry: `DATE("Expiry Date") - DATE('{today.isoformat()}')`
2. Total Inventory: `SUM("Received Packages" - "Shipped Packages")`

## FINAL INSTRUCTION:
Generate a single, complete PostgreSQL query that returns the `"Trial Name"`, `"Batch/Lot Number"`, `"Expiry Date"`, and Days Until Expiry for all lots where Days Until Expiry is less than or equal to {expiry_threshold_days} AND Total Inventory is greater than 0.
"""

    elif stage == "SHORTFALL_PREDICTION":
        # Logic from the original code (Shortfall Prediction)
        weeks_threshold = 8
        demand_cols = ['Sep', 'Oct', 'Nov'] # Example dynamic columns
        
        return f"""
You are the Supply Watchdog.
**CONTEXT:** Today is {today.isoformat()}. The lookback window for demand is the average of the last 3 months: **{demand_cols}**.
The alert threshold is stock running out in **{weeks_threshold} weeks** or less.
Your ONLY output must be the complete PostgreSQL SQL query.

## SCHEMA ABSTRACTION
| Business Concept | Required Table | Column Names (Use these exact names!) |
| :--- | :--- | :--- |
| **Supply** | `available_inventory_report` (S) | `"Trial Name"`, `"Location"`, `"Received Packages"`, `"Shipped Packages"` |
| **Demand** | `enrollment_rate_report` (D) | `"Trial Alias"`, `"Country"`, {', '.join(f'"{c}"' for c in demand_cols)}, `"Year"` |
| **Linkage** | `allocated_materials_to_orders` (M) | `trial_alias`, `trial_alias_description` (maps to S."Trial Name") |

## CALCULATION LOGIC:
1. Available Stock: SUM("Received Packages" - "Shipped Packages")
2. Monthly Demand: ({' + '.join(f'COALESCE("{c}",0)' for c in demand_cols)}) / {len(demand_cols)}.0
3. Weeks of Coverage: (Available Stock / NULLIF(Monthly Demand, 0)) * 4

## FINAL INSTRUCTION:
Generate a single, complete PostgreSQL query that identifies all combinations of Trial/Country where 'Weeks of Coverage' is less than {weeks_threshold}. Use CTEs for clarity.
"""
    return ""


# --- 7. BUILD THE GRAPH ---
def build_watchdog_graph():
    builder = StateGraph(AgentState)

    # Nodes (shared by both operations)
    builder.add_node("generate_query", generate_query_node)
    builder.add_node("execute_query", execute_query_node)
    builder.add_node("prepare_retry", prepare_retry_message)
    builder.add_node("process_results", process_results_and_advance)

    # 7.1. EXPIRY ALERT FLOW (START -> Execute -> Process)
    builder.add_edge(START, "generate_query")
    
    # Main loop for query execution
    builder.add_edge("generate_query", "execute_query")
    builder.add_conditional_edges(
        "execute_query",
        check_execution_status,
        {
            "SUCCESS": "process_results",
            "RETRY": "prepare_retry",
            "MAX_RETRIES_REACHED": "process_results", # Advance with error
        }
    )
    builder.add_edge("prepare_retry", "generate_query")

    # 7.2. TRANSITION TO SHORTFALL OR END
    builder.add_conditional_edges(
        "process_results",
        check_workflow_stage,
        {
            "ADVANCE": "generate_query", # Transition to Shortfall Prediction stage
            "END": END,
        }
    )

    return builder.compile()

# --- 8. EXECUTION ---
def run_watchdog_agent():
    """Runs the full two-stage Watchdog agent."""
    
    agent_graph = build_watchdog_graph()
    
    print(f"--- STARTING WATCHDOG AGENT (Full Workflow) ---")
    # Initial state for the Expiry Alert stage
    initial_state = {
        "messages": [
            SystemMessage(content=generate_system_prompt("EXPIRY_ALERT")),
            HumanMessage(content="Generate the SQL query for the EXPIRY ALERT now, strictly following the System Prompt and Schema.")
        ],
        "retry_count": 0,
             "max_retries": 3,
        "final_report": {"report_date": datetime.date.today().isoformat()},
        "current_operation": "EXPIRY_ALERT",
        "query_result": None,
        "error": None
    }
    
    # config = {"configurable": {"thread_id": "watchdog_session"}}
    
    # Run the graph and collect the final state
    final_state = agent_graph.invoke(initial_state)

    final_report = final_state.get("final_report", {})
    
    print("\n" + "="*50)
    print("      WATCHDOG AGENT FINAL REPORT")
    print("="*50)
    
    expiry_alerts = final_report.get("expiry_alerts", [])
    shortfall_predictions = final_report.get("shortfall_predictions", [])
    
    # Summary
    print(f"Date: {final_report.get('report_date')}")
    print(f"Expiry Alerts Found: {len(expiry_alerts)}")
    print(f"Shortfall Risks Found: {len(shortfall_predictions)}")
    print("="*50)
    
    # Full JSON Output
    print(json.dumps(final_report, indent=2, default=str))


if __name__ == "__main__":
    run_watchdog_agent()