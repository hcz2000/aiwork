import queue
import threading
import os
import functools
import gradio as gr
from playground.actions_manager import ActionsManager
from playground.assistants_api import api
from playground.assistants_panel import assistants_panel
from playground.assistants_utils import EventHandler, get_tools
from playground.btree_runner_panel import btree_runner_panel
from playground.environment_manager import EnvironmentManager
from playground.logging import Logger
from playground.semantic_manager import SemanticManager
from playground.global_values import GlobalValues
import argparse
# === Patch Gradio 6.x / Starlette 1.x TemplateResponse 兼容性问题 ===
try:
    import starlette.templating as _st_templating

    # Patch 1: Jinja2Templates.TemplateResponse 方法
    # Gradio 6.x 调用约定: TemplateResponse(template_name, context_dict, ...)
    # Starlette 1.x 期望: TemplateResponse(request, template_name, context_dict, ...)
    # 本补丁将旧版调用转换为新版调用
    _OrigJinja2Templates = getattr(_st_templating, 'Jinja2Templates', None)
    if _OrigJinja2Templates is not None:
        _orig_JT_TemplateResponse = getattr(_OrigJinja2Templates, 'TemplateResponse', None)
        if _orig_JT_TemplateResponse is not None:
            @functools.wraps(_orig_JT_TemplateResponse)
            def _patched_JT_TemplateResponse(self, *args, **kwargs):
                try:
                    # 检测旧版调用: TemplateResponse(template_name_str, context_dict, ...)
                    if (args and len(args) >= 2
                            and isinstance(args[0], str)
                            and isinstance(args[1], dict)):
                        # 旧版调用约定: (name, context, status_code, headers, media_type, background)
                        name, context = args[0], args[1]
                        remaining_args = args[2:]
                        request = context.get("request")

                        if request is not None:
                            # 新版签名: TemplateResponse(request, name, context, **kwargs)
                            return _orig_JT_TemplateResponse(self, request, name, context, *remaining_args, **kwargs)

                        raise ValueError("TemplateResponse 旧版调用缺少 request 对象，请检查 context 字典是否包含 'request' 键。")
                    # 非旧版调用，直接调用原方法
                    return _orig_JT_TemplateResponse(self, *args, **kwargs)
                except Exception as e:
                    # 兜底：如果补丁处理失败，返回简单 HTML 错误页面
                    print(f"[ERROR] TemplateResponse patch failed: {e}")
                    from starlette.responses import HTMLResponse
                    return HTMLResponse(
                        content=f"<html><body><h1>Gradio 模板渲染错误</h1><p>错误信息: {e}</p><p>请检查 Gradio 和 Starlette 版本兼容性。</p></body></html>",
                        status_code=500
                    )

            _OrigJinja2Templates.TemplateResponse = _patched_JT_TemplateResponse

except Exception as _patch_err:
    print(f"[WARN] Starlette TemplateResponse patch failed: {_patch_err}")


# Patch: gradio_client JSON Schema bug (Gradio 6.x 在处理复杂/非标准 JSON Schema的BUG)
try:
    import gradio_client.utils as _gc_utils
    _orig_js2py = getattr(_gc_utils, "_json_schema_to_python_type", None)
    if _orig_js2py is not None:
        def _patched_js2py(schema, defs=None):
            try:
                return _orig_js2py(schema, defs)
            except (TypeError, RecursionError, KeyError, AttributeError):
                return "any"
        _gc_utils._json_schema_to_python_type = _patched_js2py
except Exception as _patch_err3:
    print(f"[WARN] gradio_client schema patch failed: {_patch_err3}")

# 抑制 gr.Dropdown choices 不匹配的警告（assistant_id 可能尚未加载到 choices 列表中）
import warnings
warnings.filterwarnings(
    "ignore",
    message="The value passed into gr.Dropdown\\(\\) is not in the list of choices",
    category=UserWarning,
    module="gradio"
)
# === End patch ===

# thread = api.create_thread()  # create a new thread everytime this is run
actions_manager = ActionsManager()
semantic_manager = SemanticManager()

# create code environment
env_manager = EnvironmentManager()
env_manager.install_requirements()

logger = Logger("logs.txt")
logger.reset_logs()

current_thread = None


def create_new_thread():
    global current_thread
    current_thread = api.create_thread()
    logger.reset_logs()
    return []

create_new_thread()

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def ask_assistant(assistant_id, history, message):
    # 先确保 history 不为 None，避免后续 append 崩溃
    if history is None:
        history = []

    assistant = api.retrieve_assistant(assistant_id)
    if assistant is None:
        history.append((None, "Assistant not found."))
        return history, gr.update(value=None, interactive=False)

    attachments = []
    content = ""
    try:
        tools, actions = get_tools(assistant.tools)
    except Exception as e:
        # get_tools 可能因 tool 格式不兼容而失败，兜底返回空列表
        print(f"[WARN] get_tools failed: {e}, using empty tools")
        tools = []
        actions = []
    for file in message["files"]:
        if "Code Interpreter" in tools:
            # upload files to the thread
            file = api.upload_file(file)
            attachments += [
                {"file_id": file.id, "tools": [{"type": "code_interpreter"}]}
            ]
        else:
            with open(file, "r") as f:
                file_content = f.read()
            file = os.path.basename(file)
            file_path = os.path.join(GlobalValues.ASSISTANTS_WORKING_FOLDER, file)
            with open(file_path, "w") as fh:
                fh.write(file_content)
            history.append((f"file {file}, saved to working folder.", None))

    if message["text"] is not None:
        history.append((message["text"], None))
        content = message["text"]
    if content or attachments:  # only create a message if there is content
        api.create_thread_message(
            current_thread.id, "user", content, attachments=attachments
        )

    return history, gr.update(value=None, interactive=False)


def get_file_path(file):
    if os.path.isabs(file):
        return file

    file_path = os.path.join(GlobalValues.ASSISTANTS_WORKING_FOLDER, file)
    return file_path


def run(history, assistant_id):
    if current_thread is None:
        history.append((None, "Please create a new thread first."))
        yield history
        return

    assistant = api.retrieve_assistant(assistant_id)
    output_queue = queue.Queue()
    eh = EventHandler(output_queue)

    if assistant is None:
        msg = "Assistant not found."
        history.append((None, msg))
        yield history
        return

    def stream_worker(assistant_id, thread_id, event_handler):
        with api.run_stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=event_handler,
        ) as stream:
            for text in stream.text_deltas:
                output_queue.put(("text", text))

    # Start the initial stream
    thread_id = current_thread.id
    initial_thread = threading.Thread(
        target=stream_worker, args=(assistant.id, thread_id, eh)
    )
    initial_thread.start()
    history[-1][1] = ""
    while initial_thread.is_alive() or not output_queue.empty():
        try:
            item_type, item_value = output_queue.get(timeout=0.1)
            if item_type == "text":
                history[-1][1] += item_value
            yield history

        except queue.Empty:
            pass
    # history[-1][1] = wrap_latex_with_markdown(history[-1][1])
    yield history

    # Final flush of images
    while len(eh.images) > 0:
        file = eh.images.pop()
        file_path = get_file_path(file)
        if os.path.exists(file_path):
            history.append((None, file_path))
            history.append((None, (file_path,)))
        yield history

    initial_thread.join()
    return None


def main_interface():
    parser = argparse.ArgumentParser(description="GPT Assistants Playground")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument(
        "--in-browser", action="store_true", help="Open in browser", default=True
    )
    parser.add_argument("--show-error", action="store_true", help="Show error messages")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Custom CSS
    custom_css = """
    :root {
        --adjustment-ratio: 150px; /* Height to subtract from the viewport for chatbot */
        --iframe-ratio: 150px; /* Height of the iframe */
    }

    body, html {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100%;
    }

    #chatbot {
        height: calc(100vh - var(--adjustment-ratio)) !important; /* Uses adjustment ratio */
        overflow-y: auto !important;
    }

    #instructions textarea {
        min-height: calc(100vh - (var(--adjustment-ratio) + 575px)); /* Additional subtraction to account for other elements */
        max-height: 1000px;
        resize: vertical;
        overflow-y: auto;
    }
    #assistant_logs textarea {
        height: calc(100vh - (var(--adjustment-ratio) - 25px)); /* Initial height calculation */
        min-height: 150px; /* Set a reasonable min-height */
        max-height: 1000px; /* Max height of 1000px */
        resize: vertical; /* Allow vertical resizing only */
        overflow-y: auto; /* Enable vertical scrollbar when content exceeds max-height */
        box-sizing: border-box; /* Ensure padding and border are included in the height calculation */
    }

    #actionsnew {
        color: #000000;
    }

    #actions {
        color: #000000;
    }

    video {
        width: 300px;  /* initial width */
        height: 200px; /* initial height */
        transition: width 0.5s ease, height 0.5s ease;
        cursor: pointer;
    }
    video:hover {
        width: auto;
        height: auto;
        max-width: 100%; /* ensures it doesn't exceed the container's width */
    }

    #prefect {
        height: calc(100vh - var(--iframe-ratio)) !important; /* Uses adjustment ratio */
        overflow-y: auto !important;
    }
    """

    prefect_url = "http://127.0.0.1:4200/dashboard"

    iframe_html = f'<iframe id="prefect" src="{prefect_url}" width="100%" height="600px"></iframe>'

    theme = gr.themes.Default()

    # theme = gr.themes.Glass()
    # theme = gr.themes.Monochrome()
    # theme = gr.themes.Soft()
    # theme = "gstaff/sketch"

    with gr.Blocks(
        css=custom_css, theme=theme, title="GPT Assistants Playground"
    ) as demo:
        with gr.Tab(label="Playground"):
            with gr.Row():
                with gr.Column(scale=4):
                    assistant_id = assistants_panel(actions_manager)

                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=True,
                        container=True,
                        avatar_images=["avatar1.png", "avatar2_lite.png"],
                        layout="panel",
                        type="tuples",
                    )
                    with gr.Row():
                        with gr.Column(scale=12):
                            chat_input = gr.MultimodalTextbox(
                                interactive=True,
                                file_types=["image"],
                                placeholder="Enter message or upload file...",
                                show_label=False,
                                elem_id="chat_input",
                            )
                        with gr.Column(scale=1):
                            new_thread_button = gr.Button(
                                "Create New Thread", size="lg"
                            )

                new_thread_button.click(create_new_thread, [], [chatbot])

                chat_msg = chat_input.submit(
                    ask_assistant,
                    [assistant_id, chatbot, chat_input],
                    [chatbot, chat_input],
                )
                bot_msg = chat_msg.then(
                    run,
                    [chatbot, assistant_id],
                    [chatbot],
                )
                bot_msg.then(
                    lambda: gr.update(interactive=True),
                    None,
                    [chat_input],
                )

                chatbot.like(print_like_dislike, None, None)

        with gr.Tab(label="Agentic Behavior Tree Runner"):
            with gr.Column(scale=4):
                btree_runner = btree_runner_panel()

        with gr.Tab(label="Prefect Flow Runner"):
            gr.HTML(iframe_html)

        with gr.Tab(label="Logs"):
            with gr.Column(scale=4):
                # Add logs
                logs = gr.Code(
                    label="",
                    language="python",
                    interactive=False,
                    container=True,
                    lines=45,
                )
                demo.load(logger.read_logs, None, logs, every=1)
    demo.queue()
    # demo.launch(share=True, inbrowser=True)
    try:
        demo.launch(
            share=args.share,
            inbrowser=args.in_browser,
            show_error=args.show_error,
            debug=args.debug,
                    )
    except Exception as e:
        print(f"\n[ERROR] Gradio demo launch failed: {e}")
        print("[HINT] 如果错误与 Gradio/Starlette 模板渲染有关，请尝试:")
        print("  pip install gradio==5.25.0 gradio-client==1.3.0  (降级到 Gradio 5.x)")
        print("  或者:")
        print("  pip install starlette==0.49.3  (降级 Starlette 到 0.49.x)")
        raise


if __name__ == "__main__":
    main_interface()
    # use the following to launch in browser with a shareable link
