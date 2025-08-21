import streamlit as st
from langchain_core.messages import HumanMessage
from multi_agent_workflow import graph

st.set_page_config(page_title="🤖 MultiAgent Studio", page_icon="🤖", layout="centered")

st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1.5em;">
        <span style="font-size: 2.5rem; margin-right: 0.5em;">🤖</span>
        <span style="font-size: 2.2rem; font-weight: bold; color: #4F8BF9; letter-spacing: 1px;">MultiAgent Studio</span>
    </div>
    <div style="text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 2em;">
        <em>Ask a question and let your AI team research, analyze, and summarize for you!</em>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("input_form"):
    user_input = st.text_area("Enter your task or question:", "", height=100)
    submitted = st.form_submit_button("🚀 Run Multi-Agent Workflow")

if submitted:
    if user_input.strip():
        with st.spinner("🤖 Agents are working on your request..."):
            response = graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "assignment": user_input,
                "task_complete": False,
                "history_log": []
            })
        st.success("Done! See your results below.")
        st.markdown(
            f"""
            <div style='background: #111; border-radius: 10px; padding: 1.5em; margin-top: 1em; box-shadow: 0 2px 8px #222;'>
                <h3 style='color: #fff;'>📝 Output</h3>
                <div style='font-size: 1.1rem; color: #fff;'>
                    {response.get("illustrations", "No output generated.")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a task or question to proceed.")
