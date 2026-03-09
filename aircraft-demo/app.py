import cv2
import numpy as np
import os
import gradio as gr
 
from rag_pipeline import RAGEngine
from llm_chat import ask_llm
 
 
reference_folder = "reference_parts"
 
reference_images = {}
 
for file in os.listdir(reference_folder):
 
    path = os.path.join(reference_folder, file)
    name = os.path.splitext(file)[0]
 
    img = cv2.imread(path, 0)
    reference_images[name] = img
 
 
rag = RAGEngine()
 
current_part = None
 
 
def match_part(uploaded_image):
 
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
 
    sift = cv2.SIFT_create()
 
    kp1, des1 = sift.detectAndCompute(gray, None)
 
    bf = cv2.BFMatcher()
 
    best_match = None
    best_score = 0
    best_kp_img = None
 
    for part_name, ref_img in reference_images.items():
 
        kp2, des2 = sift.detectAndCompute(ref_img, None)
 
        matches = bf.knnMatch(des1, des2, k=2)
 
        good = []
 
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
 
        score = len(good)
 
        if score > best_score:
 
            best_score = score
            best_match = part_name
 
            best_kp_img = cv2.drawKeypoints(
                uploaded_image,
                kp1,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
 
    return best_match, best_kp_img
 
 
def process_image(image):
 
    global current_part
 
    if image is None:
        return None, "Upload an image"
 
    part, kp_image = match_part(image)
 
    if part is None:
        return kp_image, "No part matched"
 
    current_part = part
 
    stored = rag.store_manual(part)
 
    if stored:
        status = f"Matched Part: {part} | Technical manual uploaded."
    else:
        status = f"Matched Part: {part} | Manual not found"
 
    return kp_image, status

def reset_session():
    global current_part
 
    current_part = None
 
    return (
        None,   # input image
        None,   # keypoints image
        "",     # status
        []      # chatbot history
    )
 
 
def undo_detection(image):
    """
    Removes detected keypoints but keeps uploaded image.
    """
    return (
        image,  
        None,   
        "Detection undone"
    )
 
 
 
def chat(question, history):
 
    global current_part
 
    if history is None:
        history = []
 
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})
 

    yield "", history, history
 
    for updated_history in ask_llm(current_part, question, history):
        yield "", updated_history, updated_history
 
 
 
 
css = """
body {
    background-color: #f5f7fb;
}
 
#left-panel {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e5e7eb;
}
 
#chat-panel {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e5e7eb;
}
 
.gr-chatbot {
    background-color: #f9fafb;
    border-radius: 10px;
}
 
textarea {
    border-radius: 10px !important;
}
"""
 
 
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
 
    gr.Markdown(
        "# Partsign"
    )
 
    with gr.Row():
 
        with gr.Column(scale=1):
 
            gr.Markdown()
 
            input_image = gr.Image(
                type="numpy",
                label="Upload Image"
            )
 
            output_image = gr.Image(
                label="Detected Keypoints"
            )
 
            status = gr.Textbox(
                label="Match Status"
            )
            match_btn = gr.Button("Match Part", variant="primary")
            undo_btn = gr.Button("Undo Detection")
            reset_btn = gr.Button("Reset Session")
 
 
 
        # RIGHT SIDE
        with gr.Column(scale=2):
 
            gr.Markdown("### Manual Assistant")
 
            chatbot = gr.Chatbot(
                height=420,
            )
 
            msg = gr.Textbox(
                label="Ask a question about the manual",
                placeholder="Example: What is the displacement of the engine?"
            )
 


    msg.submit(
        chat,
        [msg, chatbot],
        [msg, chatbot, chatbot]
    )

    undo_btn.click(
        fn=undo_detection,
        inputs=input_image,
        outputs=[input_image, output_image, status]
    )

    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[input_image, output_image, status, chatbot]
    )

    match_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, status]
    )
 
 
demo.launch()
 