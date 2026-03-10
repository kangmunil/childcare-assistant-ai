import unittest
import unittest.mock
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.chatbot_api import app, get_agent


class StatefulAgent:
    def __init__(self):
        self.calls = []

    async def achat(
        self,
        user_input,
        chat_history=None,
        profile_context=None,
        intent_hint=None,
        growth_context=None,
        requested_profile_domains=None
    ):
        self.calls.append(
            {
                "user_input": user_input,
                "chat_history": chat_history,
                "profile_context": profile_context,
                "intent_hint": intent_hint,
                "growth_context": growth_context,
                "requested_profile_domains": requested_profile_domains,
            }
        )
        return f"응답 {len(self.calls)}"


class FakeSessionManager:
    def __init__(self):
        self.messages = {}

    def get_history(self, session_id, limit=10):
        return list(self.messages.get(session_id, []))

    def add_message(self, session_id, role, content):
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].append({"role": role, "content": content})


class TestChatbotApiSessionContinuity(unittest.TestCase):
    def setUp(self):
        self.auth_enabled_patch = patch("src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False)
        self.token_patch = patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "test-token")
        self.auth_enabled_patch.start()
        self.token_patch.start()

        self.agent = StatefulAgent()
        app.dependency_overrides[get_agent] = lambda: self.agent
        self.client = TestClient(app)
        self.session_manager = FakeSessionManager()

    def tearDown(self):
        app.dependency_overrides.clear()
        self.auth_enabled_patch.stop()
        self.token_patch.stop()

    def test_two_turn_chat_uses_same_session_history(self):
        with unittest.mock.patch("src.api.chatbot_api.session_manager", self.session_manager):
            first = self.client.post("/chat", json={"message": "안녕하세요"})
            first_payload = first.json()

            second = self.client.post(
                "/chat",
                json={
                    "message": "어제 이야기한 내용 이어서 말해줄래",
                    "session_id": first_payload["session_id"]
                }
            )
            second_payload = second.json()

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first_payload["session_id"], second_payload["session_id"])
        self.assertEqual(len(self.agent.calls), 2)
        self.assertEqual(len(self.agent.calls[0]["chat_history"]), 0)
        self.assertEqual(len(self.agent.calls[1]["chat_history"]), 2)

    def test_api_accepts_mixed_history_payload_from_session(self):
        fake_session_manager = unittest.mock.MagicMock()
        fake_session_manager.get_history.return_value = [
            {"role": "system", "content": "초기 컨텍스트"},
            {"role": "assistant", "content": "좋아요"},
            "bad-entry",
        ]
        fake_session_manager.add_message.side_effect = self.session_manager.add_message

        with unittest.mock.patch("src.api.chatbot_api.session_manager", fake_session_manager):
            response = self.client.post(
                "/chat",
                json={"message": "두 번째 질문", "session_id": "session-1"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.agent.calls), 1)
        last_call = self.agent.calls[-1]
        self.assertEqual(len(last_call["chat_history"]), 3)
        self.assertEqual(last_call["chat_history"][0]["role"], "system")
        self.assertEqual(last_call["chat_history"][1]["role"], "assistant")
        self.assertEqual(last_call["chat_history"][2], "bad-entry")


if __name__ == "__main__":
    unittest.main()
