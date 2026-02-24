import unittest

from langchain.schema import AIMessage, HumanMessage

from src.rag.childcare_agent import ChildcareAgent


class TestChildcareAgentChatHistory(unittest.TestCase):
    def setUp(self):
        self.agent = ChildcareAgent.__new__(ChildcareAgent)

    def test_history_dict_is_normalized_to_langchain_messages(self):
        history = [
            {"role": "user", "content": "안녕하세요"},
            {"role": "assistant", "content": "반가워요"},
        ]

        normalized = self.agent._to_langchain_messages(history)

        self.assertEqual(len(normalized), 2)
        self.assertIsInstance(normalized[0], HumanMessage)
        self.assertIsInstance(normalized[1], AIMessage)
        self.assertEqual(normalized[0].content, "안녕하세요")
        self.assertEqual(normalized[1].content, "반가워요")

    def test_history_with_unknown_role_falls_back_to_human_message(self):
        history = [
            {"role": "system", "content": "컨텍스트"},
            {"role": "", "content": "빈 role"},
        ]

        normalized = self.agent._to_langchain_messages(history)

        self.assertEqual(len(normalized), 2)
        self.assertIsInstance(normalized[0], HumanMessage)
        self.assertIsInstance(normalized[1], HumanMessage)
        self.assertEqual(normalized[0].content, "컨텍스트")
        self.assertEqual(normalized[1].content, "빈 role")

    def test_history_with_empty_or_invalid_entry_is_ignored(self):
        history = [
            {"role": "user", "content": ""},
            None,
            {"foo": "bar"},
            "not dict",
            {"role": "assistant", "content": "응답"},
        ]

        normalized = self.agent._to_langchain_messages(history)

        self.assertEqual(len(normalized), 1)
        self.assertIsInstance(normalized[0], AIMessage)
        self.assertEqual(normalized[0].content, "응답")

    def test_requested_profile_domains_are_normalized_and_applied_to_prompt_context(self):
        normalized = self.agent._normalize_requested_profile_domains(["sleep", "sleep", "INVALID", "medical", "", None, "allergy"])
        self.assertEqual(normalized, ["sleep", "medical", "allergy"])

        profile_context = self.agent._build_profile_context(
            "[자녀 프로필 - 신뢰 데이터]\n- 수면: 20:00~06:00",
            ["sleep", "medical"]
        )
        self.assertIn("요청 도메인: 수면, 건강", profile_context)
        self.assertIn("[자녀 프로필 - 신뢰 데이터]", profile_context)


if __name__ == "__main__":
    unittest.main()
