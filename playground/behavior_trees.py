import os

import py_trees

from playground.assistants_api import api

USE_PREFECT = os.environ.get("BTREE_USE_PREFECT", "0") == "1"

if USE_PREFECT:
    from prefect import flow


# Define the FunctionWrapper class
class FunctionWrapper:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.function(*self.args, **self.kwargs)


def resolve_assistant(assistant_name):
    """按名字取已安装的 assistant；库里没有同名助手时直接抛异常。"""
    assistant = api.get_assistant_by_name(assistant_name)
    if assistant is None:
        raise ValueError(
            "未找到名为 '%s' 的 assistant，请先在面板 Add Assistant 中创建同名助手。"
            % assistant_name
        )
    return assistant


def create_task(action):
    def init_wrapper():
        print(f"Initializing task: {action.name}")
        print(f"Assistant: {action.assistant_name}")

        action.long_running_process()

    if not USE_PREFECT:
        # 默认路径：纯函数包装，全程不 import prefect
        return init_wrapper

    # 仅在 BTREE_USE_PREFECT=1 时才走 prefect 装饰（flow 已在模块顶部按开关导入）
    return flow(
        name=action.name,
        description=action.assistant_name,
        log_prints=True,
    )(init_wrapper)


# Define the ActionWithThread class
class ActionWrapper(py_trees.behaviour.Behaviour):
    def __init__(self, name, assistant_name, function_wrapper, is_condition=False):
        super(ActionWrapper, self).__init__(name=name)
        self.assistant_name = assistant_name
        self.thread = None
        self.thread_running = False
        self.thread_success = False
        self.function_wrapper = function_wrapper
        self.is_condition = is_condition
        self.run_context = None

    def setup(self):
        # This is called once at the beginning to setup any necessary state or resources
        print("%s.setup()" % self.name)
        return py_trees.common.Status.SUCCESS

    def initialise(self):
        create_task(self)()

    def long_running_process(self):
        try:
            print("%s: Thread started, running process..." % self.name)
            result = self.function_wrapper()
            print(result)
            if "FAILURE" in result["text"]:
                print("%s: Thread completed with failure." % self.name)
                return

            if self.is_condition:
                self.thread_success = "SUCCESS" in result["text"]
            else:
                self.thread_success = True

            print("%s: Thread completed successfully." % self.name)
        except Exception as e:
            print("%s: Exception in thread: %s" % (self.name, str(e)))
        finally:
            self.thread_running = False

    def update(self):
        # This is called every tick to update the status of the behavior
        print("%s.update()" % self.name)
        if self.thread_running:
            return py_trees.common.Status.RUNNING
        else:
            return (
                py_trees.common.Status.SUCCESS
                if self.thread_success
                else py_trees.common.Status.FAILURE
            )

    def terminate(self, new_status):
        # This is called once each time the behavior terminates
        print("%s.terminate(%s)" % (self.name, new_status))
        if self.thread is not None:
            self.thread.join()
        self.thread = None
        self.thread_running = False
        self.thread_success = False


def create_assistant_action(action_name, assistant_name, assistant_instructions):
    assistant = resolve_assistant(assistant_name)
    function_wrapper = FunctionWrapper(
        api.call_assistant, assistant.id, assistant_instructions
    )
    return ActionWrapper(
        name=action_name,
        assistant_name=assistant_name,
        function_wrapper=function_wrapper,
    )


def create_assistant_condition(condition_name, assistant_name, assistant_instructions):
    assistant = resolve_assistant(assistant_name)
    function_wrapper = FunctionWrapper(
        api.call_assistant, assistant.id, assistant_instructions
    )
    return ActionWrapper(
        name=condition_name,
        assistant_name=assistant_name,
        function_wrapper=function_wrapper,
        is_condition=True,
    )


def create_assistant_action_on_thread(
    thread, action_name, assistant_name, assistant_instructions
):
    assistant = resolve_assistant(assistant_name)
    function_wrapper = FunctionWrapper(
        api.call_assistant_with_thread, thread, assistant.id, assistant_instructions
    )
    return ActionWrapper(
        name=action_name,
        assistant_name=assistant_name,
        function_wrapper=function_wrapper,
    )


def create_assistant_condition_on_thread(
    thread, condition_name, assistant_name, assistant_instructions
):
    assistant = resolve_assistant(assistant_name)
    function_wrapper = FunctionWrapper(
        api.call_assistant_with_thread, thread, assistant.id, assistant_instructions
    )
    return ActionWrapper(
        name=condition_name,
        assistant_name=assistant_name,
        function_wrapper=function_wrapper,
        is_condition=True,
    )
