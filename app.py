import gradio as gr
from xray_engine import VisionEngine
from brain_engine import BrainEngine
from rag_engine import RAGEngine
from llm_handler import LLMHandler
import whisper
 
xray_tool = VisionEngine()
brain_tool = BrainEngine()
rag_tool = RAGEngine()
llm_tool = LLMHandler()
 
whisper_model = whisper.load_model("base")
 
def transcribe_audio(audio):
    if audio is None:
        return ""
 
    result = whisper_model.transcribe(audio)
    return result["text"].strip()
 
 
def process_case(image, mode, pdf_file, state):
    overlay = None
    clean_results = {}
    blood_table = ""
    
    if image:
        if mode == "Chest X-RAY":
            results, heatmap, label, score = xray_tool.process_image(image)
            overlay = xray_tool.overlay_heatmap(image, heatmap)
 
        else:
            results, pil_img, top_idx = brain_tool.classify(image)
            overlay = brain_tool.gradcam(pil_img, top_idx)
 
        processed_results = {k.strip(): v for k, v in results.items() if k.strip() != ""}   
        top_key = max(processed_results, key=processed_results.get)
        
        significant_findings = {
            k: v for k, v in processed_results.items()
            if (v > 0.10 or k == top_key)
        }
        
        state['imaging_context'] = "SCAN RESULTS:\n" + "\n".join(
            [f"- {k}: {v:.2%} probability" for k, v in significant_findings.items()]
        )
        clean_results = processed_results
    
    if pdf_file:
        rag_tool.initialize_session(pdf_file.name)
        blood_table = rag_tool.get_blood_table(pdf_file.name)
    
    print(clean_results)
    
    return overlay, clean_results, state, blood_table
 
 
def chat_interface(audio_in, text_in, state):
    query = text_in
    if not query:
        yield "Waiting for query...", None
        return
 
    imaging_context = state.get("imaging_context", "No imaging findings.")
    blood_context = rag_tool.query_case(query)
 
    partial_answer = ""
 
    for token in llm_tool.generate_response_stream(
        query,
        imaging_context,
        blood_context
    ):
        partial_answer += token
        yield partial_answer, None   
 
    audio = llm_tool.text_to_speech(partial_answer)
    yield partial_answer, audio
 
def stream_text(text_in, state):
    """Handles only the text generation and streaming."""
    query = text_in
    if not query:
        yield "", state
        return
 
    imaging_context = state.get("imaging_context", "No imaging findings.")
    blood_context = rag_tool.query_case(query)
    history_buffer = state.get("chat_history_buffer", "")
 
    partial_answer = ""
    for token in llm_tool.generate_response_stream(
        query, imaging_context, blood_context, history=history_buffer
    ):
        partial_answer += token
        yield partial_answer, state
 
def play_audio(text_output):
    """Generates audio only after the text is fully displayed."""
    if not text_output or text_output.strip() == "":
        return None
    audio_path = llm_tool.text_to_speech(text_output)
    return audio_path
 
 
css = """
#centered_table_container { margin-top: 20px; }
#centered_table_container table { margin-left: auto; margin-right: auto; border-collapse: collapse; }
#centered_table_container p { text-align: center; }
"""
 
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    state = gr.State({})
    
    gr.HTML("<h1 style='text-align: center;'>🩺 MedAssist</h1>")
 
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["Chest X-RAY", "Brain MRI"], label="Modality", value="Brain MRI")
            img_in = gr.Image(type="filepath", label="Upload Scan")
            pdf_in = gr.File(label="Upload Blood Report (PDF)")
            audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Consult")
            text_in = gr.Textbox(label="What's your concern?", placeholder="Speak or type...", lines=3)
            submit_btn = gr.Button("RUN ANALYSIS", variant="primary")
 
        with gr.Column(scale=2):
            with gr.Row():
                cam_out = gr.Image(label="Localized Finding")
                label_out = gr.Label(label="Disease predictions")
 
            output_md = gr.Markdown("### Clinical summary")
            output_audio = gr.Audio(
                label="Audio Response",
                autoplay=True,
                interactive=False,
                show_label=False,
                container=False
            )
 
    with gr.Column(elem_id="centered_table_container"):
        gr.HTML("<hr><h3 style='text-align: center;'>📋 Major Blood Report Readings</h3>")
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=4):
                blood_report_table = gr.Markdown("*Table will appear here after PDF upload.*")
            with gr.Column(scale=1): pass
 
    img_in.upload(
        process_case,
        inputs=[img_in, mode, pdf_in, state],
        outputs=[cam_out, label_out, state, blood_report_table]
    )
    
    pdf_in.upload(
        process_case,
        inputs=[img_in, mode, pdf_in, state],
        outputs=[cam_out, label_out, state, blood_report_table]
    )
    
    audio_in.change(fn=transcribe_audio, inputs=audio_in, outputs=text_in)
 
    submit_btn.click(
        fn=lambda: (None, "Analyzing..."),
        outputs=[output_audio, output_md]
    ).then(
        fn=stream_text,
        inputs=[text_in, state],
        outputs=[output_md, state]
    ).then(
        fn=play_audio,
        inputs=[output_md],
        outputs=[output_audio]
    )
 
demo.launch()
