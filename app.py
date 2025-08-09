import streamlit as st
import torch
import cv2
import io
import os
import json
import csv
import re
import time
import tempfile
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from detect_and_localize import detect_multiple_stars, load_models
from PIL import Image
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory


st.set_page_config(layout="wide")

st.title("Star Detection & Brightness Estimation")

def _init_state():
    ss = st.session_state
    ss.setdefault("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ss.setdefault("classifier_path", "star_classifier.pth")
    ss.setdefault("regressor_path", "best_model.pth")
    ss.setdefault("models_loaded", False)
    ss.setdefault("classifier", None)
    ss.setdefault("regressor",None)
    ss.setdefault("image_path",None)
    ss.setdefault("last_params", {"prob_thr": 0.2, "stride": 8, "patch_size": 32})
    ss.setdefault("last_detections", [])
    ss.setdefault("csv_path", None)
    ss.setdefault("marked_image_path", None)

_init_state()

def _ensure_models_loaded():
    if not st.session_state.models_loaded:
        classifier, regressor = load_models(
            st.session_state.device,
            classifier_path=st.session_state.classifier_path,
            regressor_path=st.session_state.regressor_path,
        )
        st.session_state.classifier = classifier
        st.session_state.regressor = regressor
        st.session_state.models_loaded = True




def _draw_detections_on_image(image_path: str, detections: List[Dict[str, Any]], out_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    for i, det in enumerate(detections):
        x,y = det["pos"]
        center = (int(round(x)), int(round(y)))
        cv2.circle(img, center, radius=5, color=(0, 255, 255), thickness=1)
        cv2.putText(
            img, f"Y{i+1}", (center[0] + 5, center[1] - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
            color=(0, 255, 255), thickness=1
        )
    cv2.imwrite(out_path, img)
    return out_path

def explain_the_system(_:str)->str:
    return (
        "Pipeline:\n"
        "1) Split image into patches (patch_size=32, stride=K)\n"
        "2) Classifier → P(star) per patch; filter by prob_thr\n"
        "3) Regressor → (dx, dy) normalized; star center = patch_center + offset\n"
        "4) Brightness = mean over a small circular neighborhood\n"
        "5) Optional: merge near detections (duplicate suppression)\n"
    )
def _parse_params(text:str):
    try:
        obj = json.loads(text)
        return float(obj.get("prob_thr", 0.2)), int(obj.get("stride", 8))
    except Exception:
        pass
    m_thr = re.search(r"prob_thr\s*=\s*([0-9.]+)", text)
    m_str = re.search(r"stride\s*=\s*(\d+)", text)
    prob_thr = float(m_thr.group(1)) if m_thr else 0.2
    stride   = int(m_str.group(1)) if m_str else 8
    return prob_thr, stride

def rerun_the_model(action_input: str):
    if not st.session_state.image_path:
         return "No image loaded. Please upload an image first."
    _ensure_models_loaded()
    prob_thr, stride = _parse_params(action_input)
    patch_size = st.session_state.last_params.get("patch_size", 32)
    stars = detect_multiple_stars(
        st.session_state.image_path,
        st.session_state.device,
        st.session_state.classifier,
        st.session_state.regressor,
        patch_size=patch_size,
        stride=stride,
        prob_thr=prob_thr,
    )
    st.session_state.last_detections = stars
    st.session_state.last_params = {"prob_thr": prob_thr, "stride": stride, "patch_size": patch_size}

    out_path = os.path.join(tempfile.gettempdir(), f"detections_{int(time.time())}.png")
    marked = _draw_detections_on_image(st.session_state.image_path, stars, out_path)
    st.session_state.marked_image_path = marked if marked else None
    return f"Re-run completed. Found {len(stars)} stars with prob_thr={prob_thr}, stride={stride}."


def export_results(_: str):
    if not st.session_state.last_detections:
        return "No detections to export."
    csv_path = os.path.join(tempfile.gettempdir(), f"stars_{int(time.time())}.csv")
    rows = []
    for det in st.session_state.last_detections:
        (sx, sy) = det["pos"]
        (dx, dy) = det["offset"]
        (cx, cy) = det["center"]
        rows.append({
            "star_x": sx, "star_y": sy, "dx": dx, "dy": dy,
            "center_x": cx, "center_y": cy,
            "brightness": det.get("brightness", np.nan),
            "prob": det.get("prob", np.nan),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    st.session_state.csv_path = csv_path
    return f"Exported CSV: {csv_path}"


TOOLS = [
    Tool.from_function(
        name="Explain the System",
        description="Explain how the star detection pipeline works.",
        func=explain_the_system,
    ),
    Tool.from_function(
        name="Rerun the Model",
        description="Rerun detection with new params. Input like 'prob_thr=0.15, stride=8' or JSON.",
        func=rerun_the_model,
    ),
    Tool.from_function(
        name="Export the Results",
        description="Export last detections to CSV.",
        func=export_results,
    ),
]

tool_names = ", ".join([t.name for t in TOOLS])
tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in TOOLS)

REACT_PREFIX = """
You are the star-detection assistant. Use the tools when appropriate.

TOOLS:
------
{tools}

Tool usage format:
Thought: Do I need to use a tool? Yes/No
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

From previous chats:
{chat_history}

When finishing the response, use:
Thought: I now know the answer
Final Answer: <short answer>

New input:
{input}
{agent_scratchpad}
""".strip()

agent_prompt = PromptTemplate.from_template(REACT_PREFIX).partial(
    tools=tool_descriptions, tool_names=tool_names
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)



qa_template = """
You are a helpful assistant inside a star-detection app.
Use the context to answer clearly but briefly. If the user asks to change threshold/stride or to export,
you may suggest the tool phrasing (e.g., 'Rerun the Model: prob_thr=0.15, stride=8' or 'Export the Results').

Context:
- Image loaded: {has_image}
- Last params: prob_thr={prob_thr}, stride={stride}, patch_size={patch_size}
- Detections: count={count}, mean_brightness={mean_brightness}

User: {question}
Assistant:
""".strip()

qa_prompt = PromptTemplate.from_template(qa_template)

def _qa_context() -> dict:
    has_img = bool(st.session_state.image_path)
    params = st.session_state.last_params
    dets = st.session_state.last_detections or []
    mb = float(np.mean([d.get("brightness", np.nan) for d in dets if not np.isnan(d.get("brightness", np.nan))])) if dets else float("nan")
    return {
        "has_image": str(has_img),
        "prob_thr": params.get("prob_thr", 0.2),
        "stride": params.get("stride", 8),
        "patch_size": params.get("patch_size", 32),
        "count": len(dets),
        "mean_brightness": "N/A" if np.isnan(mb) else f"{mb:.4f}",
    }

def qa_answer(user_text: str) -> str:
    ctx = _qa_context()
    msg = qa_prompt.format(question=user_text, **ctx)
    return qa_llm.invoke(msg).content

_agent = create_react_agent(llm, TOOLS, agent_prompt)
_agent_executor = AgentExecutor(
    agent=_agent,
    tools=TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    early_stopping_method="generate",
)

_memory_store: Dict[str, ChatMessageHistory] = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ChatMessageHistory()
    return _memory_store[session_id]

chat_agent = RunnableWithMessageHistory(
    _agent_executor,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

    
def agent_answer(user_text: str, session_id: str) -> str:
    result = chat_agent.invoke(
        {"input": user_text},
        {"configurable": {"session_id": session_id}},
    )
    return result.get("output", str(result))



st.header("Star Detection And ChatBot")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Upload Image And Run")
    up = st.file_uploader("Upload a star image", type=["png", "jpg", "jpeg"])
    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(up.getvalue())
            st.session_state.image_path = tmp.name
        st.success("Image loaded.")
    
    prob_thr = st.slider("Classifier threshold", 0.0, 1.0, float(st.session_state.last_params["prob_thr"]), 0.01)
    stride   = st.slider("Stride (px)", 4, 32, int(st.session_state.last_params["stride"]), 4)
    patch_sz = st.selectbox("Patch size", [32], index=0, help="Currently fixed to 32 in the pipeline.")

    if st.button("Run detection"):
        if not st.session_state.image_path:
            st.error("Upload an image first.")
        else:
            _ensure_models_loaded()
            stars = detect_multiple_stars(
                st.session_state.image_path,
                st.session_state.device,
                st.session_state.classifier,
                st.session_state.regressor,
                patch_size=patch_sz,
                stride=int(stride),
                prob_thr=float(prob_thr),
            )
            st.session_state.last_detections = stars
            st.session_state.last_params = {"prob_thr": float(prob_thr), "stride": int(stride), "patch_size": int(patch_sz)}
            out_path = os.path.join(tempfile.gettempdir(), f"detections_{int(time.time())}.png")
            marked = _draw_detections_on_image(st.session_state.image_path, stars, out_path)
            st.session_state.marked_image_path = marked if marked else None
            st.success(f"Found {len(stars)} stars.")
    if st.session_state.marked_image_path:
        st.image(st.session_state.marked_image_path, caption="Marked detections", use_column_width=True)
with col_right:
    st.subheader("Detections / Export")
    if st.session_state.last_detections:
        df = pd.DataFrame([{
            "x": d["pos"][0], "y": d["pos"][1],
            "dx": d["offset"][0], "dy": d["offset"][1],
            "center_x": d["center"][0], "center_y": d["center"][1],
            "brightness": d.get("brightness", np.nan),
            "prob": d.get("prob", np.nan)
        } for d in st.session_state.last_detections])
        st.dataframe(df, use_container_width=True)
        if st.button("Export CSV"):
            msg = export_results("")
            st.info(msg)
            if st.session_state.csv_path:
                st.download_button(
                    "Download CSV",
                    data=open(st.session_state.csv_path, "rb").read(),
                    file_name=os.path.basename(st.session_state.csv_path),
                    mime="text/csv"
                )
    else:
        st.info("No detections yet.")


st.markdown("---")
st.subheader("Chat")


if "session_id" not in st.session_state:
    st.session_state.session_id = "session-001"
if "messages" not in st.session_state:
    st.session_state.messages = []


chat_mode = st.radio("Chat mode", ["Agent (tools)", "QA (no tools)"], horizontal=True)

for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
user_msg = st.chat_input("Ask about the pipeline, rerun (prob_thr/stride), export, or general questions.")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if chat_mode == "Agent (tools)":
                    try:
                        reply = agent_answer(user_msg, st.session_state.session_id)
                    except Exception as e:
                        reply = qa_answer(user_msg)
                else:
                    reply = qa_answer(user_msg)
            except Exception as e:
                reply = f"Error: {e}"

        
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})














    









    





    


    



    

 
