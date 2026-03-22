"""
OpenClaw adapter for TAU-2 benchmark evaluation.

Routes TAU-2 agent turns through OpenClaw's /v1/responses HTTP API
with native client-side function tools (gateway must be running).
"""

import json
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


def _build_system_prompt(domain_policy: str) -> str:
    """Build a concise system context for the agent (tools are provided natively)."""
    return f"""\
You are a customer service representative for a benchmark evaluation.
Help customers strictly according to the policy below.

You have two kinds of tools:
1. Your built-in OpenClaw tools (memory, file read/write, etc.) — use these freely and silently.
2. Business system tools — these appear as function tools. Call them when needed.

Rules for business system tools:
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


def _call_responses_api(user_id: str, input_data, tools: list[dict] = None, timeout: int = 300) -> dict:
    """Call OpenClaw /v1/responses endpoint and return parsed JSON response."""
    url = f"{GATEWAY_BASE_URL}/v1/responses"
    payload = {
        "model": "openclaw",
        "input": input_data,
        "stream": False,
        "user": user_id,
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _call_openclaw_simple(session_id: str, message: str, use_gateway: bool = False) -> str:
    """Simple text-only call for seeding/feedback (no tools needed)."""
    if use_gateway:
        resp = _call_responses_api(session_id, message)
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
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"openclaw agent failed: {result.stderr}")
        return result.stdout.strip()


# Keep _call_openclaw as alias for backward compatibility with run_train.py
def _call_openclaw(session_id: str, message: str, use_gateway: bool = False) -> str:
    return _call_openclaw_simple(session_id, message, use_gateway=use_gateway)


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

    def __init__(self, tools: list[Tool], domain_policy: str, use_gateway: bool = True, session_collector: Optional[list] = None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._client_tools = _tools_to_client_tools(tools)
        self._system_prompt = _build_system_prompt(domain_policy)
        self._use_gateway = use_gateway
        self._session_collector = session_collector

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> OpenClawAgentState:
        session_id = f"tau2-{uuid.uuid4().hex[:12]}"
        # Seed the session with the system context
        try:
            seed_msg = f"[SYSTEM CONTEXT]\n{self._system_prompt}"
            _call_openclaw_simple(session_id, seed_msg, use_gateway=self._use_gateway)
            print(f"  [OpenClaw] session: {session_id}")
            logger.info(f"[OpenClaw] Session {session_id} initialized")
        except Exception as e:
            logger.warning(f"[OpenClaw] Failed to seed session: {e}")
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
        elif isinstance(message, ToolMessage):
            # Send function_call_output back to the session
            call_id = state.pending_call_id or message.tool_call_id or "unknown"
            content = message.content or ""
            if message.error:
                content = f"[ERROR] {content}"
            input_data = [
                {"type": "function_call_output", "call_id": call_id, "output": content}
            ]
            state.pending_call_id = None
        elif isinstance(message, MultiToolMessage):
            # Multiple tool results — send all as function_call_output items
            input_items = []
            for tm in message.tool_messages:
                call_id = tm.tool_call_id or "unknown"
                content = tm.content or ""
                if tm.error:
                    content = f"[ERROR] {content}"
                input_items.append(
                    {"type": "function_call_output", "call_id": call_id, "output": content}
                )
            input_data = input_items
            state.pending_call_id = None
        else:
            input_data = str(message)

        logger.debug(f"[OpenClaw] Turn {state.turn_count}: sending to session {state.session_id}")
        resp = _call_responses_api(
            user_id=state.session_id,
            input_data=input_data,
            tools=self._client_tools,
        )
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
