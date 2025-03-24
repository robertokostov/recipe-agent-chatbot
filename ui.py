# ui.py
"""Defines the Gradio user interface for the Recipe Chat Agent."""

import gradio as gr
import time # Keep for respond function simulation if needed
import logging

# Local Imports
# Assuming recipe_system.py is in the same directory
from recipe_system import RecipeRecommendationSystem
from logger_setup import get_logger
from config import DEFAULT_SAMPLE_SIZE # Import default if needed

logger = get_logger()

# ==============================================================================
# Gradio Interface Creation Function
# ==============================================================================
def create_interface(recipe_system_instance: RecipeRecommendationSystem): # Accept instance
    """Sets up and defines the Gradio web interface using a stateful gr.Chatbot."""
    logger.info("Creating Gradio interface definition...")

    # --- UI Helper Functions (Operate on the passed-in instance) ---
    def ui_init_system(sample_size_value, progress=gr.Progress(track_tqdm=True)):
        logger.info(f"UI: Init clicked. Sample size: {sample_size_value}")
        status_msg = "Initializing..."
        # Outputs: Status, Init Btn, Reload Btn, Send Btn, Msg Input
        yield status_msg, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        try:
            # Use the instance passed to create_interface
            success = recipe_system_instance.initialize(force_reload=False, sample_size=int(sample_size_value))
            if success and recipe_system_instance.is_initialized:
                num = len(recipe_system_instance.recipes_df) if recipe_system_instance.recipes_df is not None else 0
                db = "vector" if recipe_system_instance.use_vector_search else "text"
                llm = "active" if recipe_system_instance.use_llm and recipe_system_instance.lc_llm else "inactive"
                status_msg = f"✅ Initialized ({num} recipes, {db} search, LLM {llm}). Ready."
                yield status_msg, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
            else:
                status_msg = f"❌ Init failed: {recipe_system_instance.initialization_error}. May use backups."
                ok = recipe_system_instance.recipes_df is not None and not recipe_system_instance.recipes_df.empty
                yield status_msg, gr.update(interactive=True), gr.update(interactive=ok), gr.update(interactive=ok), gr.update(interactive=ok)
        except Exception as e:
            logger.exception(f"UI initialization error: {e}")
            yield f"❌ UI Error: {e}", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def ui_reload_system(sample_size_value, progress=gr.Progress(track_tqdm=True)):
        logger.info(f"UI: Reload clicked. Sample size: {sample_size_value}")
        status_msg = "Reloading..."
        yield status_msg, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        try:
            # Use the instance passed to create_interface
            success = recipe_system_instance.initialize(force_reload=True, sample_size=int(sample_size_value))
            if success and recipe_system_instance.is_initialized:
                num = len(recipe_system_instance.recipes_df) if recipe_system_instance.recipes_df is not None else 0
                db = "vector" if recipe_system_instance.use_vector_search else "text"
                llm = "active" if recipe_system_instance.use_llm and recipe_system_instance.lc_llm else "inactive"
                status_msg = f"✅ Reloaded ({num} recipes, {db} search, LLM {llm}). Ready."
                yield status_msg, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
            else:
                status_msg = f"❌ Reload failed: {recipe_system_instance.initialization_error}. May use backups."
                ok = recipe_system_instance.recipes_df is not None and not recipe_system_instance.recipes_df.empty
                yield status_msg, gr.update(interactive=True), gr.update(interactive=ok), gr.update(interactive=ok), gr.update(interactive=ok)
        except Exception as e:
            logger.exception(f"UI reload error: {e}")
            yield f"❌ UI Error: {e}", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def respond(message, chat_history_list, num_results_value):
        logger.info(f"UI Chat: Msg='{message}', History Len={len(chat_history_list)}, N={num_results_value}")
        # Use the passed-in recipe_system_instance
        if not message or not message.strip():
            chat_history_list.append({"role": "user", "content": message})
            chat_history_list.append({"role": "assistant", "content": "⚠️ Please enter a message."})
            return chat_history_list, gr.update(value="")

        if not recipe_system_instance.is_initialized and (recipe_system_instance.recipes_df is None or recipe_system_instance.recipes_df.empty):
            chat_history_list.append({"role": "user", "content": message})
            chat_history_list.append({"role": "assistant", "content": "⚠️ System not initialized or no data loaded. Please Initialize/Reload."})
            return chat_history_list, gr.update(value="")

        chat_history_list.append({"role": "user", "content": message})
        chat_history_list.append({"role": "assistant", "content": "..."}) # Placeholder
        yield chat_history_list, "" # Yield history and clear update ''

        bot_response_content = "Error generating response."
        try:
            logger.info("Calling recipe_system.search_recipes...")
            # Use the instance passed to create_interface
            bot_response_content = recipe_system_instance.search_recipes(message, int(num_results_value))
            if not bot_response_content: bot_response_content = "😕 No specific information found."
            logger.info("Backend search successful.")
        except Exception as e:
            logger.exception(f"Error during backend search call from chat: {e}")
            bot_response_content = f"❌ Error calling backend: {e}"

        chat_history_list[-1]["content"] = bot_response_content
        yield chat_history_list, "" # Yield final history and clear update ''

    # --- UI Layout ---
    with gr.Blocks(
        title="Recipe Chat Agent",
        theme=gr.themes.Soft(primary_hue=gr.themes.colors.amber, secondary_hue=gr.themes.colors.lime),
        css=".gradio-container {max-width: 800px !important}"
    ) as demo:
        gr.Markdown("# 🍲 Recipe Chat Agent 🎉")
        gr.Markdown("### Ask questions or search for recipes conversationally!")

        # Define ALL UI Components FIRST
        with gr.Row():
            with gr.Column(scale=1):
                status_display = gr.Textbox("Status: Not initialized.", label="System Status", interactive=False, lines=2)
            with gr.Column(scale=2):
                with gr.Accordion("⚙️ Settings & Initialization", open=False):
                    # Use default from config, but allow UI override
                    sample_slider = gr.Slider(minimum=100, maximum=5000, value=DEFAULT_SAMPLE_SIZE, step=100, label="Recipes to Load/Sample", info="Affects init time/memory.")
                    results_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="# Results/Context Docs", info="For RAG context or # Text Results")
                    with gr.Row():
                        init_button = gr.Button("🚀 Initialize System", variant="secondary", size="sm") # Interactive state set by load
                        reload_button = gr.Button("🔄 Reload Data", variant="stop", size="sm") # Interactive state set by load

        with gr.Group(visible=True) as chat_interface_group: # Keep visible
            chatbot = gr.Chatbot(label="Conversation", bubble_full_width=False, height=500, type='messages') # Use 'messages' type
            chat_history = gr.State([]) # Initialize state for history list
            with gr.Row():
                msg_input = gr.Textbox(label="Your Message:", placeholder="Type your message here...", lines=1, scale=4, container=False) # Interactive state set by load
                send_button = gr.Button("✉️ Send", variant="primary", scale=1, min_width=100) # Interactive state set by load
            gr.Examples(
                examples=[
                    ["easy weeknight dinner"], ["healthy vegetarian soup"],
                    ["how long does the banana bread take to bake?"],
                    ["does the carbonara recipe use cream?"], ["супа со печурки"],
                    ["find recipes with feta and olives"]
                ],
                inputs=msg_input, label="Example Messages"
            )

        # --- Define ALL Event Listeners AFTER components ---
        init_button.click(
            fn=ui_init_system,
            inputs=[sample_slider],
            outputs=[status_display, init_button, reload_button, send_button, msg_input]
        )
        reload_button.click(
            fn=ui_reload_system,
            inputs=[sample_slider],
            outputs=[status_display, init_button, reload_button, send_button, msg_input]
        )

        send_button.click(
            fn=respond,
            inputs=[msg_input, chat_history, results_slider],
            outputs=[chatbot, msg_input] # Respond updates chatbot and clears input
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chat_history, results_slider],
            outputs=[chatbot, msg_input] # Respond updates chatbot and clears input
        )

        # Initial setup on load: Enable ONLY init_button
        def setup_load_state():
           # Return updates for: Init, Reload, Send, MsgInput
           return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        demo.load(
            fn=setup_load_state, inputs=None,
            outputs=[init_button, reload_button, send_button, msg_input]
        )

    logger.info("Gradio Interface definition complete.")
    return demo