import sqlite3
from typing import List, Dict
from loguru import logger
from src.core.config import settings

class ChatSessionManager:
    """
    SQLite 기반의 채팅 세션 관리자
    대화 히스토리를 DB에 영구 저장합니다.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._init_db()

    def _init_db(self):
        """세션 테이블 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON chat_sessions(session_id)')
                conn.commit()
        except Exception as e:
            logger.error(f"DB 초기화 오류: {e}")

    def add_message(self, session_id: str, role: str, content: str):
        """메시지 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chat_sessions (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"메시지 저장 오류: {e}")

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """특정 세션의 최근 히스토리 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT role, content FROM chat_sessions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit)
                )
                rows = cursor.fetchall()
                # 에이전트는 과거 메시지부터 읽어야 하므로 역순 정렬
                return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        except Exception as e:
            logger.error(f"히스토리 조회 오류: {e}")
            return []

    def delete_session(self, session_id: str):
        """세션 삭제"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"세션 삭제 오류: {e}")

# 싱글톤 인스턴스
session_manager = ChatSessionManager()
