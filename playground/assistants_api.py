import json
import os
import queue
import threading
import uuid
from types import SimpleNamespace

from peewee import SqliteDatabase, Model, TextField, FloatField, IntegrityError
from dotenv import load_dotenv
from playground.llms import get_llm_client

load_dotenv()

# Peewee ORM 模型
_db = SqliteDatabase("installed_assistants.db")


class AssistantModel(Model):
    id = TextField(primary_key=True)
    name = TextField()
    instructions = TextField()
    model = TextField()
    tools = TextField(null=True)
    response_format = TextField(null=True)
    temperature = FloatField(null=True)
    top_p = FloatField(null=True)

    class Meta:
        database = _db

class Assistant:
    def __init__(self, id, name, instructions, model, tools, response_format, temperature, top_p):
        self.id = id
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools
        self.response_format = response_format
        self.temperature = temperature
        self.top_p = top_p


class StreamWrapper:
    """包装 OpenAI streaming response，提供 text_deltas 迭代器。"""

    def __init__(self, stream_obj, eh):
        self._stream = stream_obj
        self._eh = eh

    def __iter__(self):
        for chunk in self._stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if self._eh and hasattr(self._eh, "on_text"):
                        try:
                            self._eh.on_text(delta.content)
                        except Exception:
                            pass
                    yield delta.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.close()
        return False

    @property
    def text_deltas(self):
        return self


class AssistantsAPI:
    """
        重构版本：使用 Chat Completions API 替代 OpenAI Assistants API。

        原始代码依赖 client.beta.* 端点（/beta/threads, /beta/assistants, /beta/runs），
        这些端点只有 OpenAI 官方服务器支持，本地模型（Ollama/vLLM/LM Studio 等）不提供。

        重构策略：
        1. 助手配置（assistants）改用内存字典存储，不再依赖 OpenAI 服务器
        2. 对话线程（threads）改用内存字典存储消息历史
        3. 核心推理改用 client.chat.completions.create() 调用
        4. 工具调用（tool calls）在 run_stream 中手动处理循环：
           - 调用 LLM，检测响应中的 tool_calls
           - 执行工具，将工具调用和结果写入线程历史
           - 重新调用 LLM（直到不再需要工具）
        5. 文件操作（upload/list/retrieve/delete）暂存为桩函数（stub）
        6. 保持所有方法签名不变，确保 main.py 等其他调用方无需修改
    """

    def __init__(self):
        self.client = get_llm_client()
        self._actions_manager = None
        # 内存存储替代 OpenAI 服务器存储
        self._assistants = {}   # assistant_id -> Assistant 对象
        self._threads = {}      # thread_id -> {"messages": [...], "assistant_id": ...}

        # 初始化 Peewee 数据库并加载已有助手数据
        _db.connect()
        _db.create_tables([AssistantModel], safe=True)
        self._load_assistants_from_db()


    def _init_db(self):
        """初始化 Peewee 数据库连接（已在上层 _db.connect 中处理，此处保留以兼容调用）。"""
        pass

    def _load_assistants_from_db(self):
        """从数据库加载已有的助手到内存（Peewee ORM）。"""
        for row in AssistantModel.select():
            assistant = Assistant(
                id=row.id, name=row.name, instructions=row.instructions, model=row.model,
                tools=json.loads(row.tools) if row.tools else None,
                response_format=json.loads(row.response_format) if row.response_format else None,
                temperature=row.temperature, top_p=row.top_p
            )
            self._assistants[assistant.id] = assistant

    def _save_assistant_to_db(self, assistant):
        """将助手保存到数据库（Peewee ORM，参考 tweeter_interface.py 的 create/update 模式）。"""
        try:
            AssistantModel.create(
                id=assistant.id,
                name=assistant.name,
                instructions=assistant.instructions,
                model=assistant.model,
                tools=json.dumps(assistant.tools, ensure_ascii=False) if assistant.tools else None,
                response_format=json.dumps(assistant.response_format, ensure_ascii=False) if assistant.response_format else None,
                temperature=assistant.temperature,
                top_p=assistant.top_p
            )
        except IntegrityError:
            # 已存在则更新（参考 tweeter_interface.py 的 update 写法）
            AssistantModel.update(
                name=assistant.name,
                instructions=assistant.instructions,
                model=assistant.model,
                tools=json.dumps(assistant.tools, ensure_ascii=False) if assistant.tools else None,
                response_format=json.dumps(assistant.response_format, ensure_ascii=False) if assistant.response_format else None,
                temperature=assistant.temperature,
                top_p=assistant.top_p
            ).where(AssistantModel.id == assistant.id).execute()

    def _delete_assistant_from_db(self, assistant_id):
        """从数据库删除助手。"""
        AssistantModel.delete().where(AssistantModel.id == assistant_id).execute()


    @property
    def actions_manager(self):
        """懒加载 ActionsManager，避免模块初始化时的循环导入。"""
        if self._actions_manager is None:
            from playground.actions_manager import ActionsManager
            self._actions_manager = ActionsManager()
        return self._actions_manager

    def create_thread(self):
        """创建新线程（对话上下文），返回带有 .id 属性的对象。"""
        thread_id = str(uuid.uuid4())
        self._threads[thread_id] = {
            "messages": [],
            "assistant_id": None,
        }

        class Thread:
            def __init__(self, tid):
                self.id = tid

        return Thread(thread_id)

    def create_thread_message(self, thread_id, role, content, attachments=None):
        """向线程中添加一条消息。"""
        if thread_id not in self._threads:
            raise ValueError(f"Thread {thread_id} not found")

        message = {"role": role, "content": content}
        self._threads[thread_id]["messages"].append(message)

        # 返回 mock 对象以保持 API 兼容
        class Message:
            def __init__(self):
                self.id = str(uuid.uuid4())

        return Message()

    def create_assistant(
        self,
        name,
        instructions,
        model,
        tools,
        actions,
        response_format,
        temperature,
        top_p,
    ):
        """创建助手（将配置存储在内存中，不再调用 OpenAI 服务器）。"""
        assistant_id = str(uuid.uuid4())
        # tools 和 actions 合并为工具列表
        all_tools = (tools or []) + (actions or [])

        assistant = Assistant(assistant_id, name, instructions, model, all_tools, response_format, temperature, top_p)
        self._assistants[assistant_id] = assistant
        # 持久化到数据库
        self._save_assistant_to_db(assistant)

        return assistant

    def _execute_tool(self, tool_name, raw_args):
        """
        通过工具名和原始 JSON 字符串执行工具（供流式拼接后调用）。

        执行工具调用，统一处理 JSON 解析、self 参数过滤和智能调用策略选择。
        """
        # 【调试】打印 LLM 返回的原始参数字符串，方便排查 JSON 解析失败问题
        print("[DEBUG] _execute_tool: tool=%s, raw_args=%s" % (tool_name, repr(raw_args)))

        tool_args = None
        try:
            tool_args = json.loads(raw_args)
            # 【调试】打印成功解析后的字典结构
            print("[DEBUG] json.loads 成功: %s" % repr(tool_args))
        except (json.JSONDecodeError, TypeError) as e:
            # 尝试修复常见格式问题
            print("[WARN] JSON解析失败! raw_args=%s" % repr(raw_args))
            print("[WARN] 异常详情: %s" % e)
            tool_args = {}

        if tool_args is None:
            tool_args = {}

        # 【关键修复】LLM 可能将 "self" 作为参数名传入（因为它看到了函数签名中的 self 参数），
        # 但 self 是 AssistantsAPI 实例引用，不是用户参数。必须移除避免冲突。
        if 'self' in tool_args:
            removed_self = tool_args.pop('self')
            print("[DEBUG] 移除 LLM 误传的 'self' 参数 (value=%s)" % repr(removed_self))

        # 【调试】打印最终传入工具的参数字典
        print("[DEBUG] 准备调用工具: name=%s, args=%s" % (tool_name, repr(tool_args)))

        # 查找工具 - 使用懒加载的 actions_manager
        action = self.actions_manager.get_action(tool_name)

        if not action:
            return f"[Error] 未找到工具: {tool_name}"

        # 安全执行工具 - 基于函数签名智能选择调用策略（替代原来的4种试错策略）
        import inspect
        from types import MethodType

        action_func = action["pointer"]

        # 【调试】打印工具函数信息
        try:
            sig = inspect.signature(action_func)
            params = list(sig.parameters.keys())
            print("[DEBUG] 工具函数签名: %s, 参数: %s" % (action_func.__name__, params))
            print("[DEBUG] 工具函数模块: %s" % getattr(action_func, '__module__', 'unknown'))
        except Exception as sig_err:
            print("[WARN] 无法获取函数签名: %s" % sig_err)
            params = []

        # 核心逻辑：检测函数签名是否包含 self 参数
        # - 如果包含 self → 是未绑定的实例方法 → 通过 getattr(self, name) 获取绑定方法调用
        # - 如果不包含 self → 是独立函数或已绑定方法 → 直接调用
        needs_self = 'self' in params
        result = None
        last_errors = []

        if needs_self:
            # 未绑定的实例方法：优先通过 getattr 获取绑定方法（Python 自动绑定 self）
            # 这比手动传 self 更可靠，因为 getattr 拿到的可能是经过装饰器包装的绑定方法
            bound_method = getattr(self, tool_name, None)
            if bound_method is not None and callable(bound_method):
                try:
                    result = bound_method(**tool_args)
                    print("[DEBUG] 成功: 通过 self.%s 绑定方法调用" % tool_name)
                except Exception as e:
                    last_errors.append("getattr绑定(%s): %s" % (type(e).__name__, e))
                    print("[WARN] getattr绑定调用失败: %s" % e)

            # 如果 getattr 方式失败（例如工具名与实例方法名不对应），尝试手动传入 self
            if result is None:
                try:
                    result = action_func(self, **tool_args)
                    print("[DEBUG] 成功: 未绑定函数，手动传入 self")
                except Exception as e:
                    last_errors.append("手动self(%s): %s" % (type(e).__name__, e))
                    print("[WARN] 手动self调用失败: %s" % e)
        else:
            # 独立函数或已绑定方法，直接调用
            try:
                result = action_func(**tool_args)
                print("[DEBUG] 成功: 直接调用（无需 self）")
            except Exception as e:
                last_errors.append("直接调用(%s): %s" % (type(e).__name__, e))
                print("[WARN] 直接调用失败: %s" % e)

        # 如果以上都失败，尝试 MethodType 绑定作为最后兜底
        if result is None:
            try:
                bound_func = MethodType(action_func, self)
                result = bound_func(**tool_args)
                print("[DEBUG] 成功: MethodType 绑定")
            except Exception as e:
                last_errors.append("MethodType(%s): %s" % (type(e).__name__, e))
                print("[WARN] MethodType绑定失败: %s" % e)

        # 所有策略都失败，返回错误
        if result is None:
            errors_str = "; ".join(last_errors)
            return "[Error] 工具执行失败: %s. 函数: %s" % (errors_str, action_func.__name__)


        # 【调试】打印返回值类型
        print("[DEBUG] 工具返回值类型: %s, 值: %s" % (type(result).__name__, repr(result)[:200]))

        # 安全序列化返回值
        return self._serialize_result(result)

    def _serialize_result(self, result):
        """
        安全序列化工具返回值。

        兼容原版 Assistants API 的富文本/图片提取逻辑，
        同时处理自定义对象（如 AssistantList）的序列化问题。
        """
        try:
            # 【特殊处理】AssistantList 对象：手动提取 data 并序列化
            result_type_name = type(result).__name__
            if result_type_name == "AssistantList" and hasattr(result, "data"):
                assistants = result.data
                if isinstance(assistants, list):
                    assistant_info = []
                    for a in assistants:
                        info = {}
                        for attr in ['id', 'name', 'model', 'instructions', 'temperature', 'top_p']:
                            if hasattr(a, attr):
                                info[attr] = getattr(a, attr)
                        assistant_info.append(info)
                    return json.dumps(assistant_info, ensure_ascii=False, indent=2)
                return str(assistants)

            # 优先检查是否包含原版 Assistants API 的 data 结构
            if hasattr(result, "data"):
                result_parts = []
                # 确保 data 是可迭代的
                data = result.data
                try:
                    for el in data:
                        if hasattr(el, "content"):
                            for c in el.content:
                                if hasattr(c, "text") and c.text:
                                    result_parts.append(c.text)
                                elif hasattr(c, "image_file"):
                                    result_parts.append("[图片文件]")
                except TypeError:
                    # data 不可迭代，直接转为字符串
                    pass
                if result_parts:
                    return "\n".join(result_parts)

            # 常规字符串直接返回
            if isinstance(result, str):
                return result

            # 尝试 JSON 序列化，失败则降级为 str()
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            # 兜底：如果 json.dumps 失败（比如自定义对象不可迭代），直接用 str()
            print("[WARN] 序列化失败(%s)，使用 str() 兜底: %s" % (e, result))
            return str(result)

    def run_stream(self, thread_id, assistant_id, event_handler):
        """
        流式运行助手（通过 Chat Completions API），支持工具调用循环。

        返回一个上下文管理器对象，其 .text_deltas 可迭代获取文本增量。

        核心逻辑：
        - 实现工具调用循环：当 LLM 响应包含 tool_calls 时，
          执行工具并将工具调用和结果写入线程历史，然后重新调用 LLM
        - 所有消息（用户消息、助手回复、工具调用、工具结果）都持久化到线程历史
        - 流式模式下，tool_calls 的 arguments 是增量传输的，必须在整个流式传输期间累积拼接，流结束后再统一执行工具
        - 使用后台线程运行循环，确保 __enter__ 立即返回，不阻塞
        """
        assistant = self._assistants.get(assistant_id)
        if not assistant:
            raise ValueError(f"Assistant {assistant_id} not found")

        thread = self._threads.get(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")

        # ===== 定义上下文管理器类 =====
        class RunStreamContext:
            def __init__(self, api_instance, thread_id, assistant_id, eh):
                self._api = api_instance
                self._thread_id = thread_id
                self._assistant_id = assistant_id
                self._eh = eh
                self._text_queue = queue.Queue()
                self._worker_thread = None
                self._done = False

            def __enter__(self):
                # 在后台线程启动工具调用循环，__enter__ 立即返回
                self._worker_thread = threading.Thread(
                    target=self._run_loop,
                    args=()
                )
                self._worker_thread.start()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # 等待后台线程完成
                if self._worker_thread:
                    self._worker_thread.join(timeout=60)
                return False

            def __iter__(self):
                return self

            def __next__(self):
                # 从内部队列读取文本，超时则结束
                try:
                    item = self._text_queue.get(timeout=30.0)
                except queue.Empty:
                    raise StopIteration
                if item is None:
                    # 结束信号
                    raise StopIteration
                return item

            @property
            def text_deltas(self):
                return self

            def _run_loop(self):
                """
                核心工具调用循环（在后台线程运行）。

                流程：
                1. 构建消息列表（system prompt + 线程历史）
                2. 调用 LLM（chat.completions.create stream=True）
                3. 收集响应中的文本和 tool_calls
                4. 如果有 tool_calls：
                   a. 执行工具
                   b. 将工具调用和结果写入线程历史
                   c. 回到步骤1重新调用 LLM
                5. 如果无 tool_calls（最终文本回复）：
                   a. 将文本写入线程历史
                   b. 发送结束信号
                   c. 退出循环
                """
                assistant = self._api._assistants.get(self._assistant_id)
                thread = self._api._threads.get(self._thread_id)

                if not assistant or not thread:
                    self._text_queue.put("[错误] 助手或线程不存在")
                    self._text_queue.put(None)
                    return

                # 工具调用循环
                while True:
                    # 构建消息列表：system prompt + 历史消息
                    messages = []
                    if assistant.instructions:
                        messages.append({"role": "system", "content": assistant.instructions})
                    messages.extend(thread["messages"])

                    # 准备 API 参数
                    kwargs = {
                        "model": assistant.model,
                        "messages": messages,
                        "stream": True,
                        "temperature": assistant.temperature or 0.7,
                        "top_p": assistant.top_p or 1.0,
                    }

                    # 传递 tools（如果定义了）
                    if assistant.tools:
                        kwargs["tools"] = assistant.tools

                    # 传递 response_format（如果定义了）
                    if assistant.response_format:
                        kwargs["response_format"] = assistant.response_format

                    # 【调试】打印 LLM 请求参数摘要
                    print("\n[DEBUG-STREAM] 准备调用 LLM API:")
                    print("[DEBUG-STREAM]   model=%s" % assistant.model)
                    print("[DEBUG-STREAM]   messages_count=%d" % len(messages))
                    print("[DEBUG-STREAM]   tools_count=%d" % (len(kwargs.get("tools", [])) if kwargs.get("tools") else 0))
                    # 打印最后一条消息的前100字符
                    if messages:
                        last_msg = messages[-1]
                        msg_preview = str(last_msg)[:100]
                        print("[DEBUG-STREAM]   last_message_preview=%s" % repr(msg_preview))

                    # 调用 Chat Completions API
                    try:
                        print("[DEBUG-STREAM] 发起 LLM 请求...")
                        stream = self._api.client.chat.completions.create(**kwargs)
                        print("[DEBUG-STREAM] LLM 请求已发出，开始接收流...")
                    except Exception as e:
                        print("[DEBUG-STREAM] !!! 流式传输异常中断: %s: %s" % (type(e).__name__, e))
                        self._text_queue.put("[LLM错误] %s" % str(e))
                        self._text_queue.put(None)
                        return

                    text_parts = []
                    accumulated_calls = {}  # tc_id -> {id, type, function: {name, arguments}}

                    print("\n[DEBUG-STREAM] 接收流式数据... ")

                    for chunk in stream:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                text_parts.append(delta.content)
                                # 实时流式输出文本给调用方
                                self._text_queue.put(delta.content)
                                # 通知 event handler（如果有的话）
                                if self._eh and hasattr(self._eh, "on_text"):
                                    try:
                                        self._eh.on_text(delta.content)
                                    except Exception:
                                        pass
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    # 使用 tc.index 而非 tc.id 作为键
                                    # tc.id 只在工具调用的第一个 chunk 中发送，后续 chunk 的 id 为 None
                                    # tc.index 在所有 chunk 中都存在，用于标识属于哪个工具调用
                                    idx = tc.index
                                    if idx not in accumulated_calls:
                                        accumulated_calls[idx] = {
                                            "id": tc.id or "",  # id 可能为 None（后续 chunk），用空字符串兜底
                                            "type": tc.type or "function",
                                            "function": {
                                                "name": tc.function.name if tc.function else "",
                                                "arguments": ""
                                            }
                                        }
                                    if tc.function:
                                        if tc.function.arguments:
                                            accumulated_calls[idx]["function"]["arguments"] += tc.function.arguments
                                            current_args = accumulated_calls[idx]["function"]["arguments"]

                    # 打印最终累积的工具调用数据
                    for tc_id, tc_data in accumulated_calls.items():
                        print("[DEBUG-STREAM] 工具调用要求 -> tc_id=%s, name=%s, args(%d字符)=%s" % (
                            tc_id, tc_data["function"]["name"],
                            len(tc_data["function"]["arguments"]),
                            repr(tc_data["function"]["arguments"])
                        ))

                    # 将累积的字典转换为支持 .function 属性访问的对象
                    def _make_tool_call_obj(data):
                        return SimpleNamespace(
                            id=data["id"],
                            type=data["type"],
                            function=SimpleNamespace(
                                name=data["function"]["name"],
                                arguments=data["function"]["arguments"]
                            )
                        )

                    tool_calls = [_make_tool_call_obj(v) for v in accumulated_calls.values()]

                    assistant_text = "".join(text_parts)

                    if tool_calls:
                        # ===== 有工具调用：执行工具并写入历史 =====
                        # 1. 将助手的文本回复（如果有）写入线程历史
                        if assistant_text:
                            thread["messages"].append({
                                "role": "assistant",
                                "content": assistant_text
                            })

                        # 2. 执行每个工具调用并写入历史
                        print("[DEBUG-STREAM] 准备执行第 %d 个工具调用" % len(tool_calls))
                        for tc in tool_calls:
                            try:
                                # _execute_tool 内部已统一处理 'self' 参数过滤，无需在此预过滤
                                raw_args = tc.function.arguments or ""
                                print("[DEBUG-STREAM] 即将调用工具: name=%s, args=%s" % (tc.function.name, repr(raw_args)))
                                print("[Tool Call] name=%s, arguments=%s" % (tc.function.name, repr(raw_args)))
                                # 使用 _execute_tool 直接传入字符串参数
                                tool_result = self._api._execute_tool(
                                    tc.function.name, raw_args
                                )
                                print("[Tool Result] %s" % tool_result)
                            except Exception as e:
                                # 【修复】工具执行异常不中断循环，记录错误到历史
                                tool_result = "[Error] 工具执行异常: %s" % str(e)
                                print("[ERROR] 工具执行异常: %s" % tool_result)

                            # 3. 将工具调用消息写入线程历史
                            # 【修复】确保 arguments 是有效的 JSON 字符串，避免 llama.cpp 解析失败
                            args_str = tc.function.arguments or ""

                            thread["messages"].append({
                                "role": "assistant",
                                "tool_calls": [{
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": args_str,
                                    }
                                }]
                            })

                            # 4. 将工具结果消息写入线程历史
                            thread["messages"].append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result
                            })

                        # 5. 继续循环：重新调用 LLM（它会看到工具结果并生成后续回复）
                        print("[DEBUG-STREAM] 工具调用完成，准备进入下一轮 LLM 请求...")
                        continue
                    else:
                        # ===== 最终回复 =====
                        # 将助手文本回复写入线程历史
                        if assistant_text:
                            thread["messages"].append({
                                "role": "assistant",
                                "content": assistant_text
                            })

                        # 【调试】无工具调用，这是最终回复
                        print("[DEBUG-STREAM] 无工具调用，收到最终文本回复(%d字符): %s" % (len(assistant_text), repr(assistant_text[:80])))

                        # 发送结束信号
                        self._text_queue.put(None)

                        # 退出循环
                        break

                # 确保线程历史已持久化
                print("[完成] 对话结束，线程历史消息数: %d" % len(thread["messages"]))

        # 创建上下文管理器实例并返回
        ctx = RunStreamContext(self, thread_id, assistant_id, event_handler)
        return ctx

    def list_assistants(self):
        """列出所有助手（从内存中读取）。"""
        class AssistantList:
            def __init__(self, data):
                self.data = data

        assistants = []
        for assistant in self._assistants.values():
            assistants.append(assistant)

        return AssistantList(assistants)

    def get_assistant_by_name(self, name):
        assistants = self.list_assistants()
        for assistant in assistants.data:
            if assistant.name and assistant.name.lower().startswith(name.lower()):
                return assistant
        return None

    def retrieve_assistant(self, assistant_id):
        """按 ID 检索助手。"""
        assistant = self._assistants.get(assistant_id)
        return assistant

    def update_assistant(
        self,
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
        """更新助手配置。"""
        if assistant_id not in self._assistants:
            print("[WARN] ... skipping update.")
            return None

        all_tools = (assistant_tools or []) + (assistant_actions or [])

        assistant = self._assistants[assistant_id]
        assistant.name = assistant_name
        assistant.instructions = assistant_instructions
        assistant.model = assistant_model
        assistant.tools = all_tools
        assistant.response_format = assistant_resformat
        assistant.temperature = assistant_temperature
        assistant.top_p = assistant_top_p

        # 持久化到数据库
        self._save_assistant_to_db(assistant)

        return assistant

    def delete_assistant(self, assistant_id):
        """删除助手。"""
        if assistant_id in self._assistants:
            del self._assistants[assistant_id]
            # 同步删除数据库记录
            self._delete_assistant_from_db(assistant_id)

    # Chat Completions API 不直接支持文件附件管理,这些方法保留仅为防止调用方报错。
    def upload_file(self, file, purpose="assistants"):
        """桩函数：文件上传不支持。"""
        class File:
            def __init__(self, fid, filename, purpose):
                self.id = fid
                self.filename = filename
                self.purpose = purpose
        return File(str(uuid.uuid4()), str(file), purpose)

    def list_files(self, purpose="assistants"):
        """桩函数：文件列表不支持。"""
        class FileList:
            def __init__(self):
                self.data = []
        return FileList()

    def retrieve_file(self, file_id):
        """桩函数：文件检索不支持。"""
        class File:
            def __init__(self, fid, filename, purpose):
                self.id = fid
                self.filename = filename
                self.purpose = purpose
        return File(file_id, "unknown", purpose)

    def delete_file(self, file_id):
        """桩函数：文件删除不支持。"""
        class Result:
            def __init__(self, fid):
                self.id = fid
                self.deleted = True
        return Result(file_id)

    def call_assistant(self, assistant_id, message):
        thread = self.create_thread()
        return self.call_assistant_with_thread(thread, assistant_id, message)

    def call_assistant_with_thread(self, thread, assistant_id, message):
        assistant = self.retrieve_assistant(assistant_id)
        output_queue = queue.Queue()

        if assistant is None:
            return {"text": "Assistant not found.", "files": []}

        # 将用户消息写入线程历史
        self.create_thread_message(thread.id, "user", message)

        def stream_worker(assistant_id, thread_id, event_handler):
            try:
                with self.run_stream(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    event_handler=event_handler,
                ) as stream:
                    # 【修复】不使用 for 循环(有2秒超时会导致提前退出)，
                    # 而是手动循环读取队列，直到 worker 线程真正结束
                    while stream._worker_thread.is_alive():
                        try:
                            item = stream._text_queue.get(timeout=2)
                            if item is None:
                                break
                            output_queue.put(("text", item))
                        except queue.Empty:
                            continue
                    # 读取 worker 线程结束后队列中剩余的文本
                    while True:
                        try:
                            item = stream._text_queue.get_nowait()
                            if item is None:
                                break
                            output_queue.put(("text", item))
                        except Exception:
                            break
            except Exception as e:
                error_msg = f"Run cancelled with error: {str(e)}"
                print(error_msg)
                output_queue.put(("text", error_msg))

        # Start the initial stream
        thread_id = thread.id
        initial_thread = threading.Thread(
            target=stream_worker, args=(assistant.id, thread_id, None)
        )
        initial_thread.start()
        reply = ""
        # 等待线程结束 + 确保队列中所有消息都被消费（避免竞态条件）
        initial_thread.join()
        # Drain remaining items from queue
        while not output_queue.empty():
            try:
                item_type, item_value = output_queue.get_nowait()
                if item_type == "text":
                    reply += item_value
            except Exception:
                break

        # 注意：助手回复已由 RunStreamContext._run_loop 写入线程历史，
        # 此处不再重复写入，避免历史中出现重复的 assistant 消息。
        return {"text": reply, "files": []}


# 【修复循环导入】使用 __getattr__ 实现 api 的懒加载
# 这样当 action 文件在模块初始化阶段尝试 import api 时，
# __getattr__ 会被调用，此时如果 api 尚未创建则创建它，
# 如果已经在创建过程中则返回已创建的实例（防止递归）。
# 关键在于：__init__ 不再导入任何外部模块（client 是懒加载属性），
# 所以不会触发 ActionsManager() -> action files -> assistants_api 的循环。
_api = None

def __getattr__(name):
    global _api
    if name == 'api':
        if _api is None:
            _api = AssistantsAPI()
        return _api
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# asss = api.list_assistants()
# for a in asss.data:
#     if a.name == "Sample Assistant":
#         api.delete_assistant(a.id)
