import streamlit as st
from code_debugger import debug_graph

st.set_page_config(
    page_title="Autonomous Code Debugger",
    page_icon="🐞",
    layout="wide"
)

st.title("🐞 Autonomous Code Debugger")
st.write("Analyze → Fix → Test → Review → Final Answer")

st.warning(
    "Demo only: code execution is restricted but is not a perfect security sandbox. "
    "Do not submit sensitive or malicious code."
)

code = st.text_area(
    "Paste your Python code here:",
    height=300,
    placeholder="def add(a, b):\n    return a - b\n\nprint(add(10, 5))"
)

if st.button("🚀 Debug Code", type="primary"):

    if not code.strip():
        st.error("Please enter some Python code.")
    else:
        with st.spinner("Agents are debugging your code..."):

            try:
                result = debug_graph.invoke({
                    "code": code,
                    "analysis": "",
                    "fixed_code": "",
                    "test_output": "",
                    "review": "",
                    "attempts": 0,
                    "final_answer": ""
                })

                st.success("Debugging completed! ✅")

                st.subheader("📋 Final Answer")
                st.write(result["final_answer"])

                st.subheader("🔍 Analysis")
                st.write(result["analysis"])

                st.subheader("🛠️ Fixed Code")
                st.code(result["fixed_code"], language="python")

                st.subheader("🧪 Test Output")
                st.code(result["test_output"])

                st.subheader("👀 Review")
                st.write(result["review"])

                st.info(f"Attempts used: {result['attempts']}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
