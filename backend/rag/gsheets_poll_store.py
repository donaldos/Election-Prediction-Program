"""Google Sheets 기반 여론조사 저장소 (읽기 전용).

Opinion_Poll 스프레드시트의 두 시트를 survey_id FK로 JOIN하여 PollEntry 리스트로 변환한다.
  - polls_meta: survey_id(PK) + date, district_id, pollster, method, ...
  - polls_candidates: survey_id(FK) + candidate, party, support
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime

from models.poll import PollEntry, PollMeta
from rag.poll_store import AbstractPollStore, PollStoreRegistry

logger = logging.getLogger(__name__)


@PollStoreRegistry.register("google_sheets")
class GoogleSheetsPollStore(AbstractPollStore):

    def __init__(
        self,
        spreadsheet_id: str | None = None,
        credentials_path: str | None = None,
        meta_sheet: str = "polls_meta",
        candidates_sheet: str = "polls_candidates",
        **kwargs,
    ) -> None:
        self._spreadsheet_id = (
            spreadsheet_id or os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "")
        )
        self._credentials_path = (
            credentials_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS", "service_account.json")
        )
        self._meta_sheet = meta_sheet
        self._candidates_sheet = candidates_sheet
        self._client = None

    def _get_spreadsheet(self):
        import gspread
        from google.oauth2.service_account import Credentials

        if self._client is None:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
            creds = Credentials.from_service_account_file(
                self._credentials_path, scopes=scopes,
            )
            self._client = gspread.authorize(creds)

        return self._client.open_by_key(self._spreadsheet_id)

    def load_all(self) -> list[PollEntry]:
        sh = self._get_spreadsheet()

        meta_rows = sh.worksheet(self._meta_sheet).get_all_records()
        candidate_rows = sh.worksheet(self._candidates_sheet).get_all_records()

        meta_map = {str(r["survey_id"]): r for r in meta_rows}

        entries: list[PollEntry] = []
        now = datetime.now()

        for row in candidate_rows:
            survey_id = str(row.get("survey_id", ""))
            candidate = str(row.get("candidate", ""))
            support = row.get("support", 0)

            if not survey_id or not candidate:
                logger.warning("필수 필드 누락 — row=%s", row)
                continue

            meta = meta_map.get(survey_id)
            if not meta:
                logger.warning("메타데이터 미발견 — survey_id=%s", survey_id)
                continue

            survey_date_str = str(meta.get("date", ""))
            district_id = str(meta.get("district_id", ""))
            pollster = str(meta.get("pollster", ""))

            if not all([survey_date_str, district_id, pollster]):
                continue

            try:
                survey_date = date.fromisoformat(survey_date_str)
            except ValueError:
                logger.warning("날짜 형식 오류 — date=%s", survey_date_str)
                continue

            entry = PollEntry(
                district_id=district_id,
                candidate=candidate,
                party=str(row.get("party", "")),
                support=float(support),
                pollster=pollster,
                survey_date=survey_date,
                created_at=now,
            )
            entry.id = self._make_id(entry)
            entries.append(entry)

        logger.info("Google Sheets 여론조사 로드 — %d건", len(entries))
        return entries

    def load_meta(self) -> list[PollMeta]:
        """polls_meta 시트의 조사 메타데이터를 반환한다."""
        sh = self._get_spreadsheet()
        meta_rows = sh.worksheet(self._meta_sheet).get_all_records()

        metas: list[PollMeta] = []
        for r in meta_rows:
            survey_date_str = str(r.get("date", ""))
            district_id = str(r.get("district_id", ""))
            pollster = str(r.get("pollster", ""))

            if not all([survey_date_str, district_id, pollster]):
                continue

            try:
                survey_date = date.fromisoformat(survey_date_str)
            except ValueError:
                continue

            publisher = str(r.get("pubulisher", ""))
            notes = f"조사수행: {publisher}" if publisher else ""

            metas.append(PollMeta(
                survey_date=survey_date,
                district_id=district_id,
                pollster=pollster,
                district_name=str(r.get("disctrict_name", "")),
                sample_size=int(r.get("sample_size", 0) or 0),
                margin_of_error=float(r.get("margin_of_error", 0) or 0),
                method=str(r.get("method", "")),
                source_url=str(r.get("url", "")),
                notes=notes,
            ))

        logger.info("Google Sheets 조사 메타 로드 — %d건", len(metas))
        return metas

    def save(self, entries: list[PollEntry]) -> list[PollEntry]:
        raise NotImplementedError(
            "Google Sheets는 스프레드시트에서 직접 관리합니다."
        )

    def delete(self, entry_id: str) -> bool:
        raise NotImplementedError(
            "Google Sheets는 스프레드시트에서 직접 관리합니다."
        )

    def delete_all(self, district_id: str | None = None) -> int:
        raise NotImplementedError(
            "Google Sheets는 스프레드시트에서 직접 관리합니다."
        )
