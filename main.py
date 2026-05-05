from langgraph.graph import StateGraph
from src.state import CArcState
from src.agents.ir.node import IRAgentNode

# Initialize the workflow
workflow = StateGraph(CArcState)

# Initialize the node class (this pre-loads the AI and Vectors)
ir_node = IRAgentNode()

# Add it to the graph
workflow.add_node("ir_agent", ir_node)
