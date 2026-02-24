import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.chatbot_api import app, get_agent


class FailingAgent:
    async def achat(
        self,
        user_input,
        chat_history=None,
        profile_context=None,
        intent_hint=None,
        growth_context=None
    ):
        raise RuntimeError("agent failed")


class TestChatbotApiFallback(unittest.TestCase):
    def setUp(self):
        self.auth_enabled_patch = patch("src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False)
        self.token_patch = patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "test-token")
        self.auth_enabled_patch.start()
        self.token_patch.start()

        self.failing_agent = FailingAgent()
        app.dependency_overrides[get_agent] = lambda: self.failing_agent
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.auth_enabled_patch.stop()
        self.token_patch.stop()

    def test_chat_returns_fallback_on_agent_error_and_keeps_session(self):
        fake_session_manager = MagicMock()
        fake_session_manager.get_history.return_value = []

        with patch("src.api.chatbot_api.session_manager", fake_session_manager):
            response = self.client.post(
                "/chat",
                json={"message": "안녕, 오늘 날씨 어때?"}
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("일시적인 오류", payload["reply"])
        self.assertIsInstance(payload["session_id"], str)

        self.assertGreaterEqual(fake_session_manager.add_message.call_count, 2)
        roles = [call.args[1] for call in fake_session_manager.add_message.call_args_list]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)


if __name__ == "__main__":
    unittest.main()
