"""
strategist_agent_py314.py
Python 3.14 compatible: LangChain 0.2+ (Pydantic v2) + LangGraph 1.x
- Three tools: technical, regulatory, logistics (call DB)
- LLM: Google Gemini (via langchain_google_genai) — replace with OpenAI if needed
- Final structured output produced via LangChain's structured output support
"""

import os
import json
import decimal
import psycopg2
from psycopg2 import extras
from typing import TypedDict, Literal
from dotenv import load_dotenv

# Pydantic v2 (Python 3.14 compatible)
from pydantic import BaseModel, Field

# LangChain 0.2+ imports (structured outputs + tools)
from langchain.tools import tool
from langchain_core.output_parsers import PydanticOutputParser


# Change import path depending on which provider integration you use.
# Here we use Google GenAI integration; if you prefer OpenAI, swap appropriately.
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# LangGraph (1.x) imports — graph-based workflow
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------
# 1) CONFIG
# ---------------------------------------------------------------------
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_DATABASE", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", 5432))
}

# ---------------------------------------------------------------------
# 2) DB helper (converts decimals -> floats for JSON)
# ---------------------------------------------------------------------
def run_sql_query(query: str) -> str:
    def _conv(obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return obj

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=extras.DictCursor) as cur:
            cur.execute(query)
            if cur.description:
                rows = [dict(r) for r in cur.fetchall()]
                return json.dumps(rows, default=_conv)
            else:
                conn.commit()
                return json.dumps({"status": "success", "message": "OK"})
    except Exception as e:
        # Return structured error as JSON string (tools must return str)
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        if conn:
            conn.close()

# ---------------------------------------------------------------------
# 3) Tools — decorated with LangChain's @tool (structured-tool friendly)
# ---------------------------------------------------------------------
@tool
def check_technical_constraints(query: str) -> str:
    """Run user-supplied SQL (or a generated SQL) against 're-evaluation' table."""
    return run_sql_query(query)

@tool
def check_regulatory_approval(query: str) -> str:
    """Check regulatory tables (rim / material_country_requirements)."""
    return run_sql_query(query)

@tool
def check_logistics_timeline(query: str) -> str:
    """Check ip_shipping_timelines_report or shipping timelines."""
    return run_sql_query(query)

TOOLS = [check_technical_constraints, check_regulatory_approval, check_logistics_timeline]

# ---------------------------------------------------------------------
# 4) Structured output schema (Pydantic v2)
# ---------------------------------------------------------------------
class StrategistDecision(BaseModel):
    decision: Literal["YES", "NO"] = Field(..., description="Final feasibility decision.")
    reasoning: str = Field(..., description="Detailed explanation citing data from tools.")
    technical_check: str = Field(..., description="Summary of technical check outputs.")
    regulatory_check: str = Field(..., description="Summary of regulatory check outputs.")
    logistical_check: str = Field(..., description="Summary of logistics check outputs.")

# Parser / format instructions for prompt injection into model
pydantic_parser = PydanticOutputParser(pydantic_object=StrategistDecision)
FORMAT_INSTRUCTIONS = pydantic_parser.get_format_instructions()

# ---------------------------------------------------------------------
# 5) LLM initialization (Google Gemini example)
# # ---------------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY")
)

# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",   # fastest, excellent reasoning
#     temperature=0.0,
#     api_key=os.getenv("GROQ_API_KEY")
# )

# Optionally bind tools if using LangChain agent tooling. In many modern flows,
# tools are surfaced to the model via the graph/tool-node and not via bind_tools.
# We'll keep tools in Graph as ToolNode; the LLM will produce tool-calls naturally.
# If your model integration supports .bind_tools(), you can call it:
# llm_with_tools = llm.bind_tools(TOOLS)   # if available

# ---------------------------------------------------------------------
# 6) Agent state type
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    # messages is a list of dict-like message objects expected by the LLM integration
    messages: list

# ---------------------------------------------------------------------
# 7) Node implementations (for LangGraph)
# ---------------------------------------------------------------------
def strategist_node(state: AgentState):
    """
    The strategist node: ask the LLM to analyze the user query and decide which tools to call.
    We'll ask the LLM to return either:
      - a "TOOL_CALL" action describing which tool to call and with what SQL
      - or a final_answer if it can produce one without tools (we still require 3 tools by rule)
    The graph's tools_condition() helper will route to the ToolNode automatically.
    """
    # Build a clear prompt that contains format instructions for tool-calls.
    messages = state["messages"]
    # Append the structured-output formatting instructions to the message so the model
    # knows how to produce the final JSON later.
    prompt_text = (
        "You are a Shelf-Life Extension Feasibility Advisor.\n\n"
        "You MUST call these three tools, in any order:\n"
        "1) check_technical_constraints(sql)\n"
        "2) check_regulatory_approval(sql)\n"
        "3) check_logistics_timeline(sql)\n\n"
        "For every tool call produce a short explanation. After calling ALL three tools, return "
        "a final JSON matching the schema described by the format instructions below.\n\n"
        "FORMAT INSTRUCTIONS:\n" + FORMAT_INSTRUCTIONS + "\n\n"
        "When you want to call a tool, respond with a single JSON object describing the action:\n"
        '{"action":"tool_call","tool":"<tool_name>","input":"<sql or input string>","note":"<why>"}\n\n'
        "If you have no more tools to call and are ready to finalize, produce the final object per the format.\n\n"
        "IMPORTANT RULES:"
"- ALWAYS return complete JSON."
"- NEVER break JSON across lines."
"- NEVER output partial SQL."
"- NEVER output trailing commas."
"- ALWAYS close all braces } and quotes "
"If JSON is not valid, you MUST regenerate valid JSON."
)
        
    # Build model input — integration-specific shape; using a simple text prompt structure.
    model_input = messages + [{"role": "user", "content": prompt_text}]
    # Use the llm to generate text. The exact method name may differ across providers;
    # modern LangChain ChatModels accept a simple 'generate' or 'invoke' style. We'll use .invoke() in this example.
    result = llm.invoke(model_input)  # provider-specific; raises if method missing
    return {"messages": [result]}

def formatter_node(state: AgentState):
    """
    After tools have been called and their outputs placed into state.messages,
    ask the model to produce a final Pydantic-validated JSON using PydanticOutputParser.
    """
    messages = state["messages"]
    # Ask model to generate final structured JSON. Provide the parser's instructions.
    final_prompt = [
        {"role": "user", "content": "Using the data above, produce the final decision JSON."},
        {"role": "user", "content": FORMAT_INSTRUCTIONS}
    ]
    model_input = messages + final_prompt
    final_resp = llm.invoke(model_input)
    # Attempt to parse model output into the pydantic model
    try:
        parsed = pydantic_parser.parse(final_resp.content)  # returns StrategistDecision
        # Convert to JSON string to keep message objects consistent
        return {"messages": [{"role": "assistant", "content": parsed.model_dump_json()}]}
    except Exception as e:
        # If parsing fails, return raw content but mark as error
        return {"messages": [{"role": "assistant", "content": json.dumps({"status": "parse_error", "raw": str(final_resp.content), "error": str(e)})}]}

# ---------------------------------------------------------------------
# 8) Build LangGraph workflow
# ---------------------------------------------------------------------
builder = StateGraph(AgentState)

# Add nodes: strategist -> tools -> strategist (loop) -> formatter -> END
builder.add_node("strategist", strategist_node)
builder.add_node("tools", ToolNode(TOOLS))        # ToolNode will surface our @tool-decorated functions
builder.add_node("formatter", formatter_node)

# Start the flow
builder.add_edge(START, "strategist")

# Use the helper that inspects the strategist output to determine whether tools should be called.
# tools_condition is a supplied helper that inspects model messages for tool_call actions.
builder.add_conditional_edges(
    "strategist",
    tools_condition,
    {"tools": "tools", "__end__": "formatter"}
)

# After tools run, go back to strategist to continue reasoning (typical ReAct loop)
builder.add_edge("tools", "strategist")

# Final formatting -> END
builder.add_edge("formatter", END)

# Memory / checkpointing (optional)
memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------
# 9) Runner (example)
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an assistant that MUST check three constraints (technical, regulatory, logistical)
by calling the corresponding tools. Always call all three tools before finalizing.
"""

def run_agent(user_query: str):
    initial_state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    }

    config = {"configurable": {"thread_id": "session_1"}}
    for event in agent_graph.stream(initial_state, config=config):
        for node_name, payload in event.items():
            print(f"\n--- Node: {node_name} ---")
            if "messages" in payload:
                last = payload["messages"][-1]
                # We assume message is provider object or dict-like
                if isinstance(last, dict):
                    print("Message (dict):", last.get("content") if last.get("content") else last)
                else:
                    # fallback: try to print `.content`
                    content = getattr(last, "content", str(last))
                    print("Message:", str(content))

if __name__ == "__main__":
    user_q = "Can we extend Batch LOT-123 for Germany? Provide SQL for checks and decide."
    run_agent(user_q)
