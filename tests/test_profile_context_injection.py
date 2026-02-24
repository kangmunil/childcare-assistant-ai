import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.chatbot_api import app, get_agent


class FakeAgent:
    def __init__(self):
        self.last_profile_context = None
        self.last_user_input = None
        self.last_intent_hint = None
        self.last_growth_context = None
        self.last_requested_profile_domains = None

    async def achat(self, user_input, chat_history=None, profile_context=None, intent_hint=None, growth_context=None, requested_profile_domains=None):
        self.last_user_input = user_input
        self.last_profile_context = profile_context
        self.last_intent_hint = intent_hint
        self.last_growth_context = growth_context
        self.last_requested_profile_domains = requested_profile_domains
        return "테스트 응답"


class TestProfileContextInjection(unittest.TestCase):
    def setUp(self):
        self.auth_enabled_patch = patch("src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False)
        self.token_patch = patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "test-token")
        self.auth_enabled_patch.start()
        self.token_patch.start()

        self.fake_agent = FakeAgent()
        app.dependency_overrides[get_agent] = lambda: self.fake_agent
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.auth_enabled_patch.stop()
        self.token_patch.stop()

    def test_chat_request_accepts_profile_context(self):
        response = self.client.post(
            "/chat",
            json={
                "message": "아이가 편식을 해요",
                "profile_context": "[자녀 프로필 - 신뢰 데이터]\n- 알레르기: 계란"
            }
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reply"], "테스트 응답")

    def test_profile_context_is_forwarded_to_agent(self):
        profile_context = "[자녀 프로필 - 신뢰 데이터]\n- 수면: 21:30~06:30"
        expected_context = "[자녀 프로필 - 신뢰 데이터] - 수면: 21:30~06:30"

        response = self.client.post(
            "/chat",
            json={
                "message": "우리 아이 수면 패턴에 맞춰 조언해줘",
                "profile_context": profile_context,
                "child_id": 10
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_agent.last_profile_context, expected_context)

    def test_requested_profile_domains_are_forwarded_to_agent(self):
        response = self.client.post(
            "/chat",
            json={
                "message": "수면 루틴이 궁금해요",
                "requested_profile_domains": ["sleep", "routine"]
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_agent.last_requested_profile_domains, ["sleep", "routine"])

    def test_manual_mode_without_profile_context_forwards_empty_context(self):
        response = self.client.post(
            "/chat",
            json={
                "message": "수동 입력 없이 답해줘",
                "context_mode": "MANUAL"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_agent.last_profile_context, "")

    def test_invalid_context_mode_returns_bad_request(self):
        response = self.client.post(
            "/chat",
            json={
                "message": "테스트",
                "context_mode": "INVALID"
            }
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["code"], "AI_003_BAD_REQUEST")

    def test_growth_fields_are_forwarded_to_agent(self):
        response = self.client.post(
            "/chat",
            json={
                "message": "아이 성장발달 확인해줘",
                "intent_hint": "growth_check",
                "growth_context": {
                    "gender": "M",
                    "birth_date": "2024-01-01",
                    "height_cm": 82.5,
                    "weight_kg": 11.2
                }
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_agent.last_intent_hint, "GROWTH_CHECK")
        self.assertEqual(self.fake_agent.last_growth_context["gender"], "M")

    def test_internal_auth_missing_token_returns_unauthorized_when_enabled(self):
        with patch("src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", True), patch(
            "src.api.chatbot_api.AI_INTERNAL_TOKEN",
            "secure-token"
        ):
            response = self.client.post(
                "/chat",
                json={
                    "message": "테스트",
                }
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["code"], "AI_005_UNAUTHORIZED")

    def test_internal_auth_rejects_invalid_token(self):
        with patch("src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", True), patch(
            "src.api.chatbot_api.AI_INTERNAL_TOKEN",
            "secure-token"
        ):
            response = self.client.post(
                "/chat",
                headers={"X-Internal-Service-Token": "wrong-token"},
                json={
                    "message": "테스트",
                }
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["code"], "AI_006_FORBIDDEN")

    def test_growth_resolution_skipped_without_internal_request(self):
        with patch("src.api.chatbot_api._resolve_growth_context_from_child") as resolve_growth:
            response = self.client.post(
                "/chat",
                json={
                    "child_id": 10,
                    "message": "성장발달 확인해줘",
                    "intent_hint": "GROWTH_CHECK",
                }
            )

        self.assertEqual(response.status_code, 200)
        resolve_growth.assert_not_called()

    def test_growth_resolution_skipped_without_matching_internal_token(self):
        with patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "secure-token"), patch(
            "src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False
        ), patch("src.api.chatbot_api._resolve_growth_context_from_child") as resolve_growth:
            response = self.client.post(
                "/chat",
                headers={"X-Internal-Service-Token": "wrong-token"},
                json={
                    "child_id": 10,
                    "message": "성장발달 확인해줘",
                    "intent_hint": "GROWTH_CHECK",
                }
            )

        self.assertEqual(response.status_code, 200)
        resolve_growth.assert_not_called()

    def test_growth_resolution_skipped_for_non_growth_intent(self):
        with patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "secure-token"), patch(
            "src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False
        ), patch("src.api.chatbot_api._resolve_growth_context_from_child") as resolve_growth:
            response = self.client.post(
                "/chat",
                headers={"X-Internal-Service-Token": "secure-token"},
                json={
                    "child_id": 10,
                    "message": "아이 수면 패턴이 너무 불규칙해요",
                    "intent_hint": "SLEEP",
                    "requested_profile_domains": ["sleep", "routine"],
                }
            )

        self.assertEqual(response.status_code, 200)
        resolve_growth.assert_not_called()
        self.assertEqual(self.fake_agent.last_intent_hint, "SLEEP")

    def test_growth_context_is_resolved_with_valid_internal_header(self):
        with patch("src.api.chatbot_api.AI_INTERNAL_TOKEN", "secure-token"), patch(
            "src.api.chatbot_api.AI_REQUIRE_INTERNAL_AUTH", False
        ), patch(
            "src.api.chatbot_api._resolve_growth_context_from_child",
            return_value=(
                {"height_cm": 92.5, "weight_kg": 11.2, "gender": "M", "birth_date": "2024-01-01"},
                {"name": "테스트"},
                None,
            ),
        ) as resolve_growth:
            response = self.client.post(
                "/chat",
                headers={"X-Internal-Service-Token": "secure-token"},
                json={
                    "child_id": 10,
                    "message": "성장발달 확인해줘",
                    "intent_hint": "GROWTH_CHECK",
                }
            )

        self.assertEqual(response.status_code, 200)
        resolve_growth.assert_called_once_with(10)
        self.assertEqual(self.fake_agent.last_growth_context["height_cm"], 92.5)

    def test_profile_context_is_sanitized_before_forward(self):
        response = self.client.post(
            "/chat",
            json={
                "message": "수면 패턴 정리해줘",
                "context_mode": "MANUAL",
                "profile_context": "  키\r\n \t무\t단   \"바꿔\"\n"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_agent.last_profile_context, '키 무단 "바꿔"')


if __name__ == "__main__":
    unittest.main()
