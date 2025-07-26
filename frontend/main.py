"""
Gradio application entry point with WebSocket integration
"""
import gradio as gr
from components.sidebar import create_sidebar
from components.search_pane import create_search_pane
from components.document_viewer import create_document_viewer
from components.admin_panel import create_admin_panel


def create_interface():
    with gr.Blocks(title="Patexia Legal AI", theme=gr.themes.Soft()) as demo:
        # Main interface with tabs
        with gr.Tabs():
            with gr.TabItem("Legal Search"):
                with gr.Row():
                    sidebar = create_sidebar()
                    search_pane = create_search_pane()
                    document_viewer = create_document_viewer()
            
            with gr.TabItem("Admin Panel"):
                admin_panel = create_admin_panel()
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )