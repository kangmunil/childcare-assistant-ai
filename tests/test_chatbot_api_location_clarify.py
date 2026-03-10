import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.chatbot_api import app, get_agent


class DummyAgent:
    async def achat(
        self,
        user_input,
        chat_history=None,
        profile_context=None,
        intent_hint=None,
        growth_context=None,
        requested_profile_domains=None,
    ):
        return "정상 응답"


class TestChatbotApiLocationClarify(unittest.TestCase):
    def setUp(self):
        self.auth_enabled_patch = patch("src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False)
        self.token_patch = patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "test-token")
        self.meta_patch = patch("src.api.chatbot_api.AI_CHAT_META_ENABLED", True)
        self.auth_enabled_patch.start()
        self.token_patch.start()
        self.meta_patch.start()

        self.agent = DummyAgent()
        app.dependency_overrides[get_agent] = lambda: self.agent
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.auth_enabled_patch.stop()
        self.token_patch.stop()
        self.meta_patch.stop()

    def test_location_search_without_region_returns_clarify_meta(self):
        fake_session_manager = MagicMock()
        fake_session_manager.get_history.return_value = []

        with patch("src.api.chatbot_api.session_manager", fake_session_manager):
            response = self.client.post("/chat", json={"message": "어린이집 찾아줘"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["meta"]["response_mode"], "CLARIFY")
        self.assertEqual(payload["meta"]["clarification"]["missing_fields"], ["location"])
        self.assertGreaterEqual(len(payload["meta"]["clarification"]["options"]), 1)
        self.assertIn("지역", payload["reply"])


if __name__ == "__main__":
    unittest.main()
