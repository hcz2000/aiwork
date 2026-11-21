
import time

import gradio as gr

from playground.assistants_api import api
from playground.assistants_utils import (
    get_actions_by_name,
    get_tools,
    get_tools_by_name,
)


def assistants_panel(actions_manager):
    available_actions = actions_manager.get_actions()

    def list_assistants():
        assistant_choices = api.list_assistants()
        assistant_options = {a.name: a.id for a in assistant_choices.data}
        assistant_options["Create New Assistant"] = "new"
        return assistant_options

    assistant_options = list_assistants()

    # 统一映射成 choices 内的值，保证「回显值 == 用户看到的值」。
    RESFORMAT_CHOICES = ("JSON object", "Plain text")

    def resformat_to_choice(value):
        """把库里存的 response_format 归一化成 Radio 的合法选项。"""
        if value in ("JSON object", "type"):
            return "JSON object"
        if value in ("Plain text", "auto"):
            return "Plain text"
        return "JSON object"

    def update_assistant(
        assistant_name,
        assistant_id,
        assistant_instructions,
        assistant_model,
        assistant_tools,
        assistant_actions,
        assistant_resformat,
        assistant_temperature,
        assistant_top_p,
    ):
        if assistant_id is None or len(assistant_id) <= 5:
            return "⚠️ 请先在上方选择一个已存在的助手，再保存修改。", gr.update()

        tools = get_tools_by_name(assistant_tools)
        actions = get_actions_by_name(assistant_actions, available_actions)

        # get_tools() 交给 UI 的是 function name，get_actions_by_name() 却按 available_action["name"] 反查.
        # 两者不一致时,会静默返回更少的项目。这里只告警、不改行为。
        if len(actions) < len(assistant_actions or []):
            print(
                "[WARN] update_assistant: 勾选了 %d 个 Action，只有 %d 个在 available_actions 里反查到，"
                "其余将被静默丢弃（疑似 action name 与 function name 不一致）: %s"
                % (len(assistant_actions or []), len(actions), assistant_actions)
            )

        # 只有用户真的动了 Radio 才写入新值，避免 "auto" 被静默改写成 "JSON object"。
        stored = api.retrieve_assistant(assistant_id)
        stored_format = stored.response_format if stored is not None else None
        resformat = (
            stored_format
            if resformat_to_choice(stored_format) == assistant_resformat
            else assistant_resformat
        )

        api.update_assistant(
            assistant_name,
            assistant_id,
            assistant_instructions,
            assistant_model,
            tools,
            actions,
            resformat,
            assistant_temperature,
            assistant_top_p,
        )

        # 下拉框的 choices 是面板构建时算好的一份快照（assistant_options），
        # 改名后不重新拉取，列表里就会一直显示旧名字。这里保存成功后重算一次，
        # 并把选中项对齐到新名字（value 必须是 name，不是 id）。
        assistant_options = list_assistants()
        if assistant_name and assistant_name in assistant_options:
            dropdown_update = gr.update(
                choices=list(assistant_options.keys()), value=assistant_name
            )
        else:
            # 新名字没进列表（空名 / 保存未生效）：只刷新候选项，不动当前选中值，
            # 否则 Gradio 会因为 value 不在 choices 里而把下拉框清空。
            dropdown_update = gr.update(choices=list(assistant_options.keys()))

        return "✅ 已保存", dropdown_update

    def create_assistant(
        assistant_name_new,
        assistant_instructions_new,
        assistant_model_new,
        assistant_tools_new,
        assistant_actions_new,
        assistant_resformat_new,
        assistant_temperature_new,
        assistant_top_p_new,
    ):
        tools = [
            {"type": "file_search"}
            if tool == "File search"
            else {"type": "code_interpreter"}
            for tool in assistant_tools_new
        ]
        actions = [
            action["agent_action"]
            for action in available_actions
            if action["name"] in assistant_actions_new
        ]
        format = "auto"  # "type" if assistant_resformat_new == "JSON object" else "auto"  TODO: fix this
        new_assistant = api.create_assistant(
            assistant_name_new,
            assistant_instructions_new,
            assistant_model_new,
            tools,
            actions,
            format,
            assistant_temperature_new,
            assistant_top_p_new,
        )
        return new_assistant.id

    def get_assistant_details(assistant_key):
        assistant_options = list_assistants()
        if assistant_key == "Create New Assistant":
            return "", "", "", "llama.cpp/Qwen3.5-9B", [], "", "", 1, 1
        else:
            if assistant_key in assistant_options.keys():
                assistant_id = assistant_options[assistant_key]
            else:
                assistant_id = assistant_key

            assistant = api.retrieve_assistant(assistant_id)
            actions = []
            if assistant.tools is None:
                tools = []
            else:
                tools, actions = get_tools(assistant.tools)
            format = resformat_to_choice(assistant.response_format)

            return (
                assistant.name,
                assistant.id,
                assistant.instructions,
                assistant.model,
                tools,
                actions,
                format,
                assistant.temperature,
                assistant.top_p,
            )

    def assistant_selected_change(assistant_key):
        if assistant_key == "Create New Assistant":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

    action_choices = sorted([action["name"] for action in available_actions])
    assistant_selected = gr.Dropdown(
        label="Select Assistant",
        choices=assistant_options.keys(),
        interactive=True,
    )

    with gr.Column(visible=False) as new_assistant_form:
        assistant_name_new = gr.Textbox(
            label="Enter a user-friendly name",
            placeholder="Enter Name",
            interactive=True,
        )
        assistant_instructions_new = gr.Textbox(
            label="Instructions",
            placeholder="You are a helpful assistant.",
            elem_id="instructions",
            interactive=True,
        )
        assistant_model_new = gr.Dropdown(
            label="Model",
            choices=["llama.cpp/Qwen3.5-9B"],
            value="llama.cpp/Qwen3.5-9B",
            interactive=True,
        )
        assistant_tools_new = gr.CheckboxGroup(
            label="Tools", choices=["File search", "Code interpreter"], interactive=True
        )
        # assistant_files_new = gr.Textbox(
        #     label="Add Files or Functions", placeholder="+ Files, + Functions"
        # )
        with gr.Accordion("Actions", open=False, elem_id="actionsnew"):
            assistant_actions_new = gr.CheckboxGroup(
                label="Actions", choices=action_choices, interactive=True
            )
        with gr.Accordion("Model settings", open=False, elem_id="modelsettingsnew"):
            assistant_resformat_new = gr.Radio(
                label="Response Format",
                choices=["JSON object", "Plain text"],
                value="JSON object",
                interactive=True,
            )
            assistant_temperature_new = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                step=0.01,
                value=1,
                interactive=True,
            )
            assistant_top_p_new = gr.Slider(
                label="Top P",
                minimum=0,
                maximum=1,
                step=0.01,
                value=1,
                interactive=True,
            )
        add_button = gr.Button("Add Assistant", interactive=True, visible=True)

    with gr.Column(visible=True) as existing_assistant_form:
        assistant_id = gr.Markdown("ID:")
        assistant_name = gr.Textbox(
            label="Enter a user-friendly name", placeholder="Enter Name"
        )
        assistant_instructions = gr.Textbox(
            label="Instructions",
            placeholder="You are a helpful assistant.",
            elem_id="instructions",
        )
        assistant_model = gr.Dropdown(
            label="Model",
            choices=["llama.cpp/Qwen3.5-9B"],
            value="llama.cpp/Qwen3.5-9B",
        )
        assistant_tools = gr.CheckboxGroup(
            label="Tools", choices=["File search", "Code interpreter"]
        )
        with gr.Accordion("Actions", open=False, elem_id="actions"):
            assistant_actions = gr.CheckboxGroup(
                label="Actions", choices=action_choices, interactive=True
            )
        with gr.Accordion("Model settings", open=False, elem_id="modelsettings"):
            assistant_resformat = gr.Radio(
                label="Response Format",
                choices=["JSON object", "Plain text"],
                value="JSON object",
            )
            assistant_temperature = gr.Slider(
                label="Temperature", minimum=0, maximum=1, step=0.01, value=1
            )
            assistant_top_p = gr.Slider(
                label="Top P", minimum=0, maximum=1, step=0.01, value=1
            )
        # Save / Delete 放在同一行：gr.Row 内两个按钮 scale 等分，各自只有原来一半宽。
        # 二者都在 existing_assistant_form 这个 Column 里，随它一起显隐，
        # 所以不会出现「一个被隐藏、另一个被撑满整行」的观感。
        with gr.Row():
            save_button = gr.Button("💾 Save", interactive=True, scale=1)
            delete_button = gr.Button("🗑️ Delete", scale=1)
        save_status = gr.Markdown("")

    assistant_selected.change(
        fn=get_assistant_details,
        inputs=assistant_selected,
        outputs=[
            assistant_name,
            assistant_id,
            assistant_instructions,
            assistant_model,
            assistant_tools,
            assistant_actions,
            assistant_resformat,
            assistant_temperature,
            assistant_top_p,
        ],
    )

    assistant_selected.change(
        fn=assistant_selected_change,
        inputs=assistant_selected,
        outputs=[
            new_assistant_form,
            existing_assistant_form,
            add_button,
            delete_button,
        ],
    )

    controls = [
        assistant_name,
        assistant_id,
        assistant_instructions,
        assistant_model,
        assistant_tools,
        assistant_actions,
        assistant_resformat,
        assistant_temperature,
        assistant_top_p,
    ]




    def create_and_select_assistant(
        assistant_name_new,
        assistant_instructions_new,
        assistant_model_new,
        assistant_tools_new,
        assistant_actions_new,
        assistant_resformat_new,
        assistant_temperature_new,
        assistant_top_p_new,
    ):
        new_assistant_id = create_assistant(
            assistant_name_new,
            assistant_instructions_new,
            assistant_model_new,
            assistant_tools_new,
            assistant_actions_new,
            assistant_resformat_new,
            assistant_temperature_new,
            assistant_top_p_new,
        )
        time.sleep(2)  # wait for assistant to be created
        assistant_options = list_assistants()
        return (
            gr.update(choices=list(assistant_options.keys()), value=new_assistant_id),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    def delete_and_select_assistant(assistant_id):
        api.delete_assistant(assistant_id)
        assistant_options = list_assistants()
        return (
            gr.update(
                choices=list(assistant_options.keys()), value="Create New Assistant"
            ),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    add_button.click(
        fn=create_and_select_assistant,
        inputs=[
            assistant_name_new,
            assistant_instructions_new,
            assistant_model_new,
            assistant_tools_new,
            assistant_actions_new,
            assistant_resformat_new,
            assistant_temperature_new,
            assistant_top_p_new,
        ],
        outputs=[
            assistant_selected,
            new_assistant_form,
            existing_assistant_form,
            add_button,
            delete_button,
        ],
    )

    # outputs 里带上 assistant_selected：改名后同步刷新下拉框列表。
    save_button.click(
        fn=update_assistant,
        inputs=controls,
        outputs=[save_status, assistant_selected],
    )

    delete_button.click(
        fn=delete_and_select_assistant,
        inputs=[assistant_id],
        outputs=[
            assistant_selected,
            new_assistant_form,
            existing_assistant_form,
            add_button,
            delete_button,
        ],
    )

    return assistant_id
