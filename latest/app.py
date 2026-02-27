import gradio as gr
import tempfile
from rag_storage import RAGEngine
import clinical_logic as cl
from brain_engine import BrainEngine
from xray_engine import VisionEngine
import time
 
brain_classifier = BrainEngine()
xray_classifier = VisionEngine()
rag_engine = RAGEngine()
 
session_results = {"brain": None, "chest": None}
 
def reset_all_sessions():
    cl.reset_clinical_memory()
 
    global session_results
    session_results = {"brain": None, "chest": None}
 
    return (
        None,           # img_in
        "Brain MRI",    # mod_in
        None,           # pdf_in
        None,             # chatbot
        "",             # msg_input
        "Session Reset Complete"
    )
 
def handle_upload(image, modality, pdf):
    status = []
    if image:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(tmp.name)
        if modality == "Brain MRI":
            preds, _, _ = brain_classifier.classify(tmp.name)
            session_results["brain"] = preds
            status.append("Brain MRI Analyzed.")
        else:
            preds, _, _, _ = xray_classifier.process_image(tmp.name)
            session_results["chest"] = preds
            status.append("Chest X-Ray Analyzed.")
    
    if pdf:
        if rag_engine.initialize_session(pdf.name):
            status.append("Blood Report Indexed.")
    
    return " | ".join(status) if status else "No files uploaded."
 
def add_user_message(history, message):
    if not message: return history, ""
    history.append({"role": "user", "content": message})
    return history, ""
 
def bot_response(history):
    raw_input = history[-1]["content"]
    
    routes, clean_user_msg = cl.route_question(raw_input)
    
    history.append({"role": "assistant", "content": ""})
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    if not routes:
        history[-1]["content"] = "I couldn't categorize that request. Could you please specify if you're asking about the MRI, X-Ray, or Blood Report?"
        yield history
        return
 
    full_ai_response = ""
    for mode, sub_question in routes.items():
        prompt = None
        if mode == "brain" and session_results.get("brain"):
            prompt = cl.get_brain_prompt(sub_question, session_results["brain"])
        elif mode == "chest" and session_results.get("chest"):
            prompt = cl.get_chest_prompt(sub_question, session_results["chest"])
        elif mode == "blood":
            context = rag_engine.query_case(sub_question)
            prompt = cl.get_blood_prompt(sub_question, context)
 
        if prompt:
            streamer = cl.get_streamer_response(prompt)
            for token in streamer:
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                token_count += 1
                history[-1]["content"] += token
                full_ai_response += token
                yield history
            history[-1]["content"] += "\n\n"
            full_ai_response += "\n\n"
    end_time = time.time()
    total_duration = end_time - start_time
    # TPS = (Total Tokens) / (Time from first token to last)
    tps = token_count / (end_time - (start_time + first_token_time)) if token_count > 1 else 0
 
    # Characterization Summary
    stats = (
        f"📊 Performance: TTFT: {first_token_time:.2f}s | "
        f"TPS: {tps:.2f} tokens/s | "
        f"Total: {total_duration:.2f}s | Tokens: {token_count}"
    )
    
    print(stats)
 
 
    cl.save_to_memory(clean_user_msg, full_ai_response)
    yield history
 
 
custom_css = """
#centered-title {
    text-align: center;
}
#reset-btn {
    background-color: #ff4b4b !important; /* Medical Red */
    color: white !important;
}
#reset-btn:hover {
    background-color: #d32f2f !important;
}
"""
with gr.Blocks(theme=gr.themes.Soft(), title="MedAssist AI") as demo:
    gr.Markdown("#MedAssist: Your clinical AI Assistant", elem_id="centered-title")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Clinical Image")
            mod_in = gr.Radio(["Brain MRI", "Chest X-Ray"], label="Modality", value="Brain MRI")
            pdf_in = gr.File(label="Blood Report (PDF)")
            process_btn = gr.Button("⚡ Process & Analyze", variant="primary")
            status_box = gr.Textbox(label="System Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Clinical Conversation", height=600)
            msg_input = gr.Textbox(label="What's your concern?", placeholder="Press Enter to send")
            reset_btn = gr.Button("🗑️ Reset Session", variant="stop", elem_id="reset-btn")
 
    msg_input.submit(
        add_user_message, [chatbot, msg_input], [chatbot, msg_input], queue=False
    ).then(
        bot_response, [chatbot], [chatbot]
    )
 
    reset_btn.click(
        fn=reset_all_sessions,
        inputs=[],
        outputs=[img_in, mod_in, pdf_in, chatbot, msg_input, status_box]
    )
    process_btn.click(handle_upload, [img_in, mod_in, pdf_in], status_box)
 
if __name__ == "__main__":
    demo.queue().launch()
 
