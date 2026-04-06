import cv2
import numpy as np
import os
import gradio as gr
 
from rag_pipeline import RAGEngine
from llm_chat import ask_llm
from session_memory import memory_manager
from speech_to_text import transcribe_audio
from gradation_logger import logger
from audio_feedback import next_generation, enqueue_sentence

 
 
reference_folder = "reference_parts"
 
reference_images = {}
 
for file in os.listdir(reference_folder):
 
    path = os.path.join(reference_folder, file)
    name = os.path.splitext(file)[0]
 
    img = cv2.imread(path, 0)
    reference_images[name] = img
 
 
rag = RAGEngine()
 
current_part = None

def get_memory_size():
 
    session = memory_manager.get_session("manual_chat")
 
    return len(session.get_messages())

def match_part(uploaded_image):
 
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
 
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray, None)

 
    bf = cv2.BFMatcher()
 
    best_match = None
    best_score = 0
    best_ref_img = None
    best_good_matches = None
    best_kp2 = None
 
    for part_name, ref_img in reference_images.items():
 
        kp2, des2 = sift.detectAndCompute(ref_img, None)
 
        if des2 is None or des1 is None:
            continue

       
        matches = bf.knnMatch(des1, des2, k=2)
        
 
        good = []
 
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
 
        score = len(good)
 
        if score > best_score:
            best_score = score
            best_match = part_name
            best_ref_img = ref_img
            best_good_matches = good
            best_kp2 = kp2
 
    keypoint_img = cv2.drawKeypoints(
        uploaded_image,
        kp1,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
 
    MIN_MATCH_COUNT = 40
 
    if best_score < MIN_MATCH_COUNT:
        return None, keypoint_img, None
 
    # RANSAC geometric verification
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([best_kp2[m.trainIdx].pt for m in best_good_matches]).reshape(-1, 1, 2)


    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

 
    if mask is None:
        return None, keypoint_img, None
 
    inliers = np.sum(mask)
 
    if inliers < 30:
        return None, keypoint_img, None
 
    return best_match, keypoint_img, best_ref_img
 
 
 
def process_image(image):
 
    global current_part
    global original_uploaded_image
    gen_id = next_generation()
 
    if image is None:
        return None, None, "Upload an image"
    
    original_uploaded_image = image.copy()
 
    part, keypoint_img, ref_img = match_part(image)
 
    if part is None:
        enqueue_sentence("No part matched.", gen_id)
        return keypoint_img, None, "No part matched", gr.update(interactive = False), gr.update(interactive = False)
    
 
    current_part = part
 
    stored = rag.store_manual(part)
 
    if stored:
        status = f"Similar part found: {part}"
        enqueue_sentence(f"Similar part found: {part}.", gen_id)
    else:
        status = f"Similar part found: {part} | Manual not found"
        enqueue_sentence(f"Similar part found: {part}. Manual not found", gen_id)
 
    return keypoint_img, ref_img, status, gr.update(interactive = True), gr.update(interactive = True)
 

def reset_session():
    global current_part

    session = memory_manager.get_session("manual_chat")

    before_clear = len(session.get_messages())

    memory_manager.clear_session("manual_chat")

    after_clear = len(session.get_messages())

    current_part = None

    memory_status = f"Memory before Reset:{before_clear} | Memory After Reset: {after_clear}"
 
    return (
        None,   # input image
        None,   # keypoints image
        "",     # status
        [],      # chatbot history
        memory_status,
        gr.update(interactive = False),
        gr.update(interactive = False)
    )
 
 
def undo_detection(image):
    global current_part
    global original_uploaded_image
    """
    Removes detected keypoints but keeps uploaded image.
    """

    current_part = None
    if image is None:
        return None, None, "No image uploaded"
    return original_uploaded_image, None, "Detection Undone"
 
 
 
# def chat(question, history):

#     logger.start_pipeline()

 
#     global current_part
 
#     if history is None:
#         history = []
 
#     history.append({"role": "user", "content": question})
#     history.append({"role": "assistant", "content": ""})
 

#     yield "", history, history
 
#     for updated_history in ask_llm(current_part, question, history):
#         yield "", updated_history, updated_history
    
#     logger.end_pipeline()



def chat(question, history):
    logger.start_pipeline()

    # ✅ stop any previous TTS immediately
    generation_id = next_generation()

    global current_part

    if history is None:
        history = []

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})

    yield "", history, history

    for updated_history in ask_llm(current_part, question, history, generation_id):
        yield "", updated_history, updated_history

    logger.end_pipeline()

 
 
 
 
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
        "# PartSign-Assist"
    )
 
    with gr.Row():
 
        with gr.Column(scale=1):
 
            gr.Markdown()
 
            input_image = gr.Image(
                type="numpy",
                label="Upload Image"
            )

            matched_part_image = gr.Image(
                label="Matched Part"
            )
 
 
            status = gr.Textbox(
                label="Match Status"
            )
            match_btn = gr.Button("Match Part", variant="primary")
            undo_btn = gr.Button("Undo Detection")
            reset_btn = gr.Button("Reset Session")
            memory_status = gr.Textbox(
                label = "Memory Debug",
                interactive = False
            )
 
 
 
        # RIGHT SIDE
        with gr.Column(scale=2):
 
 
            chatbot = gr.Chatbot(
                height=420,
            )


            msg = gr.Textbox(
                label="What's your query?",
                placeholder="Example: What is the name of the part?",
                scale=6
            )
        
            ask_btn = gr.Button("Ask", variant="primary", scale=1, interactive = False)
        
        
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Ask your question",
                show_label = True,
                scale=1,
                visible = True,
                interactive = False
            )
 

    msg.submit(
        chat,
        [msg, chatbot],
        [msg, chatbot, chatbot]
    )

    undo_btn.click(
        fn=undo_detection,
        inputs=input_image,
        outputs=[input_image, matched_part_image, status]
    )

    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[input_image, matched_part_image, status, chatbot, memory_status, ask_btn, mic]
    )

    match_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[input_image, matched_part_image, status, ask_btn, mic]
    )

    mic.stop_recording(
        fn=transcribe_audio,
        inputs=mic,
        outputs=[msg, mic]
    )

    ask_btn.click(
    chat,
    [msg, chatbot],
    [msg, chatbot, chatbot]
)

demo.launch()
 