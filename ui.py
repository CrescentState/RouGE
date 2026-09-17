import streamlit as st
import os
import sys

dag_path = os.path.join(os.path.dirname(__file__), "DAG")
sys.path.append(dag_path)

from dag_models import DagNode
from decompose import decompose, get_free_vram_mb, call_int8_engine

st.set_page_config(page_title="RouGE Unified Pipeline", layout="wide")

# ==========================================
# PHASE 1: CLASSIFICATION (Placeholder)
# ==========================================
def run_contract1_classifier(prompt: str):
    """
    TODO: When Person A finishes the classification folder, import their function here.
    For now, this provides a manual override so you can test the pipeline.
    """
    token_length = len(prompt.split())
    # Simple heuristic for testing: if it has "and" or "then", call it mixed_intent
    if " and " in prompt.lower() or " then " in prompt.lower():
        return "mixed_intent", token_length
    return "single_intent", token_length

# ==========================================
# PHASE 2: DAG VISUALIZATION HELPER
# ==========================================
def build_mermaid_chart(node, graph_lines=None, parent_id=None):
    if graph_lines is None:
        graph_lines = ["graph TD"]
    
    safe_text = node.text.replace('"', "'").replace("\n", " ")[:40] + "..."
    current_id = node.node_id
    
    if node.resolved_via == "int8_fallback":
        shape = f"{current_id}>INT8 Fallback:<br>{safe_text}]"
    elif node.resolved_via == "decomposed":
        shape = f"{current_id}{{INT4 Decomposed:<br>{safe_text}}}"
    else:
        shape = f"{current_id}[Leaf Node:<br>{safe_text}]"
        
    graph_lines.append(shape)
    
    if parent_id:
        graph_lines.append(f"{parent_id} --> {current_id}")
        
    for child in node.children:
        build_mermaid_chart(child, graph_lines, current_id)
        
    return "\n".join(graph_lines)

def extract_executable_tasks(node, task_list=None):
    """Recursively fetch all nodes that need final execution."""
    if task_list is None:
        task_list = []
    
    if node.resolved_via in ["leaf", "int8_fallback"]:
        task_list.append(node)
    
    for child in node.children:
        extract_executable_tasks(child, task_list)
        
    return task_list

# ==========================================
# UI LAYOUT & EXECUTION
# ==========================================
st.title("RouGE Architecture: End-to-End Pipeline")
st.markdown(f"**Hardware Status:** RTX 5050 | **Free VRAM:** `{get_free_vram_mb():.0f} MB`")
st.markdown("---")

prompt = st.text_area("Enter a user prompt:", height=100, placeholder="e.g., Summarize the incident report and then draft a follow-up email...")

# Manual override for testing Phase 2 until Phase 1 is integrated
intent_override = st.selectbox(
    "Force Intent Type (For Testing):", 
    ["auto", "single_intent", "mixed_intent", "length_escalated"]
)

if st.button("Execute Pipeline", type="primary"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        # --- PHASE 1 ---
        with st.status("Phase 1: Classification", expanded=True) as status:
            st.write("Routing through Contract 1...")
            auto_intent, tokens = run_contract1_classifier(prompt)
            
            final_intent = auto_intent if intent_override == "auto" else intent_override
            st.success(f"Classified as: **{final_intent}** ({tokens} tokens)")
            status.update(label="Phase 1 Complete", state="complete", expanded=False)

        # --- PHASE 2 ---
        with st.status("Phase 2: VRAM-Gated Decomposition", expanded=True) as status:
            st.write("Evaluating hardware limits and splitting tasks...")
            root = DagNode(
                node_id="eval_root", 
                text=prompt, 
                intent_type=final_intent, 
                token_length=tokens
            )
            
            tree = decompose(root, depth=0, original_full_prompt=prompt)
            
            st.markdown("### DAG Topology")
            mermaid_code = build_mermaid_chart(tree)
            st.markdown(f"```mermaid\n{mermaid_code}\n```")
            status.update(label="Phase 2 Complete", state="complete", expanded=False)

        # --- PHASE 3 ---
        st.subheader("Phase 3: Final Execution (INT8)")
        tasks = extract_executable_tasks(tree)
        
        if not tasks:
            st.info("No executable tasks found.")
        else:
            # Create a visual tab for each sub-task to keep the UI clean
            tabs = st.tabs([f"Task {i+1}" for i in range(len(tasks))])
            
            for idx, (tab, task) in enumerate(zip(tabs, tasks)):
                with tab:
                    st.markdown(f"**Instruction:** `{task.text}`")
                    with st.spinner("Generating answer..."):
                        # Send the specific sub-task to your live INT8 engine
                        answer = call_int8_engine(task.text)
                        st.markdown("**AI Response:**")
                        st.info(answer)