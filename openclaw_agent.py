"""
OpenClaw adapter for TAU-2 benchmark evaluation.

Routes TAU-2 agent turns through OpenClaw's /v1/responses HTTP API
with native client-side function tools (gateway must be running).
"""

import json
import subprocess
import uuid
from typing import Optional

import urllib.request
import urllib.error

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage, is_valid_agent_history_message
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


GATEWAY_BASE_URL = "http://127.0.0.1:18789"

DOMAIN_AGENT_MAP = {
    "airline": "tau2-airline",
    "retail": "tau2-retail",
    "telecom": "tau2-telecom",
}


def _build_system_prompt(domain_policy: str) -> str:
    """Build a concise system context for the agent (tools are provided natively)."""
    return f"""\
You are a customer service representative for a benchmark evaluation.
Help customers strictly according to the policy below.

You have two kinds of tools:
1. Your built-in OpenClaw tools (memory, file read/write, etc.) — use these freely and silently.
2. Business system tools — these appear as function tools. Call them when needed.

Rules for business system tools:
- Call ONE tool at a time. Wait for the result before calling the next tool.
- Only pass arguments the customer explicitly provided. Do NOT infer or add optional arguments.
- After receiving a tool result, use it to continue helping the customer.
- You may use your OpenClaw tools (e.g. memory) alongside business tools.

## Policy
{domain_policy}

Now wait for the customer to start. Follow the policy exactly."""


def _tools_to_client_tools(tools: list[Tool]) -> list[dict]:
    """Convert TAU-2 Tool objects to OpenAI-style function tool definitions."""
    client_tools = []
    for tool in tools:
        schema = tool.openai_schema
        fn = schema.get("function", schema)
        client_tools.append({
            "type": "function",
            "function": {
                "name": fn.get("name", tool.name),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        })
    return client_tools


class OpenClawAgentState(BaseModel):
    session_id: str
    turn_count: int = 0
    pending_call_id: Optional[str] = None
    pending_tool_call_id: Optional[str] = None
    memory_prefixed: bool = False


def _call_responses_api(user_id: str, input_data, tools: list[dict] = None, timeout: int = 300, agent_id: Optional[str] = None, instructions: Optional[str] = None) -> dict:
    """Call OpenClaw /v1/responses endpoint and return parsed JSON response."""
    url = f"{GATEWAY_BASE_URL}/v1/responses"
    payload = {
        "model": "openclaw",
        "input": input_data,
        "stream": False,
        "user": user_id,
    }
    if instructions:
        payload["instructions"] = instructions
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if agent_id:
        headers["x-openclaw-agent-id"] = agent_id
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _call_openclaw_simple(session_id: str, message: str, use_gateway: bool = False, agent_id: Optional[str] = None) -> str:
    """Simple text-only call for seeding/feedback (no tools needed)."""
    if use_gateway:
        resp = _call_responses_api(session_id, message, agent_id=agent_id)
        # Extract text from response
        parts = []
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    parts.append(c["text"])
        return " ".join(parts).strip()
    else:
        import subprocess, os
        env = os.environ.copy()
        if "VOLCANO_ENGINE_API_KEY" not in env:
            try:
                cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
                with open(cfg_path) as f:
                    cfg = json.load(f)
                api_key = cfg.get("env", {}).get("VOLCANO_ENGINE_API_KEY", "")
                if api_key:
                    env["VOLCANO_ENGINE_API_KEY"] = api_key
            except Exception:
                pass
        cmd = ["openclaw", "agent", "--local", "--json", "--session-id", session_id, "--message", message]
        if agent_id:
            cmd += ["--agent-id", agent_id]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"openclaw agent failed: {result.stderr}")
        return result.stdout.strip()


def _seed_session(session_id: str, message: str, use_gateway: bool = False, agent_id: Optional[str] = None) -> str:
    """Seed a session via CLI. When use_gateway=True, omit --local so the session
    is registered in the running gateway under the correct agent."""
    import subprocess, os
    env = os.environ.copy()
    if "VOLCANO_ENGINE_API_KEY" not in env:
        try:
            cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
            with open(cfg_path) as f:
                cfg = json.load(f)
            api_key = cfg.get("env", {}).get("VOLCANO_ENGINE_API_KEY", "")
            if api_key:
                env["VOLCANO_ENGINE_API_KEY"] = api_key
        except Exception:
            pass
    cmd = ["openclaw", "agent", "--json", "--session-id", session_id, "--message", message]
    if not use_gateway:
        cmd.insert(2, "--local")
    if agent_id:
        cmd += ["--agent-id", agent_id]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"openclaw agent seed failed: {result.stderr}")
    return result.stdout.strip()


# Keep _call_openclaw as alias for backward compatibility with run_train.py
def _call_openclaw(session_id: str, message: str, use_gateway: bool = False, agent_id: Optional[str] = None) -> str:
    return _call_openclaw_simple(session_id, message, use_gateway=use_gateway, agent_id=agent_id)


def _wrap_tool_output(content: str) -> str:
    """Ensure tool output is JSON so the memory hook's { check skips it."""
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return content
    return json.dumps({"result": content})


def _run_ov_json(args: list[str]) -> object:
    result = subprocess.run(["ov", *args], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"ov {' '.join(args)} failed")
    stdout = result.stdout.strip()
    if not stdout:
        return None
    return json.loads(stdout)


def _extract_ov_search_uris(search_result: object, limit: int = 3) -> list[str]:
    if not isinstance(search_result, dict):
        return []
    payload = search_result.get("result") if isinstance(search_result.get("result"), dict) else search_result
    uris = []
    for key in ("memories", "resources", "skills"):
        for item in payload.get(key, []) or []:
            uri = item.get("uri") if isinstance(item, dict) else None
            if uri and uri not in uris:
                uris.append(uri)
            if len(uris) >= limit:
                return uris
    return uris


def _clean_ov_read_text(text: str) -> str:
    return text.strip()


def _format_memory_fields(text: str) -> str:
    marker = "<!-- MEMORY_FIELDS"
    start = text.find(marker)
    if start == -1:
        return text.strip()

    json_start = text.find("{", start)
    json_end = text.rfind("-->")
    if json_start == -1 or json_end == -1:
        return text[:start].strip()

    try:
        payload = json.loads(text[json_start:json_end].strip())
    except json.JSONDecodeError:
        return text[:start].strip()

    lines = []
    for key, value in payload.items():
        if key == "trajectory_ids":
            continue
        if isinstance(value, str):
            lines.append(f"{key}:\n{value}")
        else:
            lines.append(f"{key}:\n{json.dumps(value, ensure_ascii=False, indent=2)}")
    return "\n\n".join(lines).strip()


def _format_openviking_memories(query: str, limit: int = 3, agent_id: Optional[str] = None) -> tuple[str, int]:
    try:
        search_args = ["search", "-o", "json", "-n", str(limit), "-u", "viking://agent", query]
        print(f"  [OpenViking] search_args={search_args[:-1]} query={query!r}")
        search_result = _run_ov_json(search_args)
        print(f"  [OpenViking] raw_search_result={json.dumps(search_result, ensure_ascii=False)[:2000]}")
        uris = _extract_ov_search_uris(search_result, limit=limit)
        print(f"  [OpenViking] search_hits={len(uris)}")
        if not uris:
            return "", 0

        memory_blocks = []
        for idx, uri in enumerate(uris, 1):
            content = subprocess.run(
                ["ov", "read", "-o", "json", uri],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if content.returncode != 0:
                print(f"  [OpenViking] read_failed uri={uri}")
                continue
            text = _format_memory_fields(_clean_ov_read_text(content.stdout))
            if not text:
                print(f"  [OpenViking] read_empty uri={uri}")
                continue
            memory_blocks.append(f"[{idx}] {uri}\n{text}")

        print(f"  [OpenViking] readable_memories={len(memory_blocks)}")
        if not memory_blocks:
            return "", 0

        return "<retrieved_memory>\n" + "\n\n".join(memory_blocks) + "\n</retrieved_memory>", len(memory_blocks)
    except Exception as e:
        logger.warning(f"[OpenClaw] OpenViking memory retrieval failed: {e}")
        return "", 0


def _parse_responses_output(resp: dict, tools: list[Tool]) -> AssistantMessage:
    """Parse /v1/responses JSON into an AssistantMessage.

    Handles two output item types:
    - function_call: the model wants to call a client tool
    - message: the model returns text
    """
    for item in resp.get("output", []):
        if item.get("type") == "function_call":
            tc = ToolCall(
                id=item.get("call_id", str(uuid.uuid4())),
                name=item["name"],
                arguments=json.loads(item.get("arguments", "{}")),
                requestor="assistant",
            )
            logger.debug(f"[OpenClaw] Tool call: {tc.name}({tc.arguments})")
            return AssistantMessage(role="assistant", tool_calls=[tc])

        if item.get("type") == "message":
            parts = []
            for c in item.get("content", []):
                if c.get("type") == "output_text" and c.get("text"):
                    parts.append(c["text"])
            text = " ".join(parts).strip()
            if text:
                logger.debug(f"[OpenClaw] Text response: {text[:100]}")
                return AssistantMessage(role="assistant", content=text)

    logger.warning(f"[OpenClaw] Could not parse response: {json.dumps(resp)[:200]}")
    return AssistantMessage(role="assistant", content="I'm here to help. Could you please clarify your request?")


class OpenClawAgent(LocalAgent[OpenClawAgentState]):
    """
    TAU-2 LocalAgent that uses OpenClaw as its backend via /v1/responses API.

    Business system tools are passed as native client-side function tools.
    OpenClaw's memory and built-in tools are available alongside.

    Gateway must be running (`openclaw gateway`).
    """

    def __init__(self, tools: list[Tool], domain_policy: str, use_gateway: bool = True, session_collector: Optional[list] = None, agent_id: Optional[str] = None, prepend_openviking_memory: bool = False):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._client_tools = _tools_to_client_tools(tools)
        self._system_prompt = _build_system_prompt(domain_policy)
        self._use_gateway = use_gateway
        self._session_collector = session_collector
        self._agent_id = agent_id
        self._prepend_openviking_memory = prepend_openviking_memory

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> OpenClawAgentState:
        session_id = f"tau2-{uuid.uuid4().hex[:12]}"
        print(f"  [OpenClaw] session: {session_id}")
        if self._session_collector is not None:
            self._session_collector.append(session_id)
        return OpenClawAgentState(session_id=session_id)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: OpenClawAgentState
    ) -> tuple[AssistantMessage, OpenClawAgentState]:
        state.turn_count += 1

        # Build the input for /v1/responses
        if isinstance(message, UserMessage):
            input_data = message.content or ""
            if self._prepend_openviking_memory and not state.memory_prefixed and input_data:
                memory_block, memory_count = _format_openviking_memories(input_data, limit=3)
                if memory_block:
                    print(f"  [OpenViking] prepended {memory_count} memories")
                    input_data = f"{memory_block}\n\n{input_data}"
                else:
                    print("  [OpenViking] no memories found")
                state.memory_prefixed = True
        elif isinstance(message, ToolMessage):
            # Send function_call_output back to the session
            call_id = state.pending_call_id or message.id or "unknown"
            content = message.content or ""
            if message.error:
                content = f"[ERROR] {content}"
            input_data = [
                {"type": "function_call_output", "call_id": call_id, "output": _wrap_tool_output(content)}
            ]
            state.pending_call_id = None
        elif isinstance(message, MultiToolMessage):
            # Multiple tool results — send all as function_call_output items
            input_items = []
            for tm in message.tool_messages:
                call_id = tm.id or "unknown"
                content = tm.content or ""
                if tm.error:
                    content = f"[ERROR] {content}"
                input_items.append(
                    {"type": "function_call_output", "call_id": call_id, "output": _wrap_tool_output(content)}
                )
            input_data = input_items
            state.pending_call_id = None
        else:
            input_data = str(message)

        logger.debug(f"[OpenClaw] Turn {state.turn_count}: sending to session {state.session_id}")
        try:
            resp = _call_responses_api(
                user_id=state.session_id,
                input_data=input_data,
                tools=self._client_tools,
                agent_id=self._agent_id,
                instructions=self._system_prompt,
            )
        except (TimeoutError, OSError) as e:
            print(f"  [OpenClaw] network error (task will fail): {e}")
            return AssistantMessage(role="assistant", content=f"[ERROR] Request failed: {e}"), state
        assistant_message = _parse_responses_output(resp, self.tools)

        # Track call_id for function_call responses so we can match the output
        if assistant_message.tool_calls:
            tc = assistant_message.tool_calls[0]
            state.pending_call_id = tc.id

        return assistant_message, state

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[OpenClawAgentState] = None,
    ) -> None:
        pass
