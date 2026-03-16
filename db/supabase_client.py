"""Supabase接続 — SQLite互換のラッパー

sqlite3.Row と同じインターフェースで Supabase (PostgreSQL) を操作できる。
既存コードの get_connection() / conn.execute() をそのまま使えるようにする。
"""

import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

_supabase: Client | None = None


def get_supabase_client() -> Client:
    """Supabaseクライアントのシングルトン"""
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL と SUPABASE_KEY を .env に設定してください"
            )
        _supabase = create_client(url, key)
    return _supabase


class SupabaseRow(dict):
    """sqlite3.Row互換: row["col"] でも row[index] でもアクセス可能"""

    def __init__(self, data: dict, columns: list[str] | None = None):
        super().__init__(data)
        self._columns = columns or list(data.keys())

    def __getitem__(self, key):
        if isinstance(key, int):
            col = self._columns[key]
            return super().__getitem__(col)
        return super().__getitem__(key)

    def keys(self):
        return self._columns


class SupabaseCursor:
    """sqlite3.Cursor互換のSupabaseラッパー"""

    def __init__(self, client: Client):
        self._client = client
        self._results: list[SupabaseRow] = []
        self._description = None

    def fetchone(self) -> SupabaseRow | None:
        return self._results[0] if self._results else None

    def fetchall(self) -> list[SupabaseRow]:
        return self._results


class SupabaseConnection:
    """sqlite3.Connection互換のSupabaseラッパー

    既存コードが conn.execute(SQL, params) で使えるように、
    SQL文をパースして Supabase REST API に変換する。
    """

    def __init__(self, client: Client):
        self._client = client

    def execute(self, sql: str, params: tuple = ()) -> SupabaseCursor:
        """SQL文をパースしてSupabase APIに変換"""
        cursor = SupabaseCursor(self._client)
        sql_stripped = sql.strip().upper()

        if sql_stripped.startswith("SELECT"):
            cursor._results = self._exec_select(sql, params)
        elif sql_stripped.startswith("INSERT"):
            self._exec_insert(sql, params)
        elif sql_stripped.startswith("CREATE") or sql_stripped.startswith("PRAGMA"):
            pass  # Supabaseではスキーマはダッシュボードで管理
        else:
            logger.debug("Skipping unsupported SQL: %s", sql[:80])

        return cursor

    def executescript(self, sql: str):
        """CREATE TABLE等のスクリプト — Supabaseでは不要"""
        pass

    def commit(self):
        """Supabaseは自動コミット"""
        pass

    def close(self):
        """接続プールはクライアントが管理"""
        pass

    def _exec_select(self, sql: str, params: tuple) -> list[SupabaseRow]:
        """SELECT文をSupabase queryに変換"""
        import re

        sql_clean = " ".join(sql.split())

        # テーブル名を抽出
        from_match = re.search(r"FROM\s+(\w+)", sql_clean, re.IGNORECASE)
        if not from_match:
            return []
        table = from_match.group(1)

        # JOIN先テーブルを確認
        join_match = re.search(
            r"JOIN\s+(\w+)\s+(\w+)\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)",
            sql_clean, re.IGNORECASE,
        )

        # SELECT カラムを取得
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_clean, re.IGNORECASE)
        select_cols = select_match.group(1).strip() if select_match else "*"

        # WHERE 条件を解析
        where_match = re.search(r"WHERE\s+(.*?)(?:ORDER|GROUP|LIMIT|$)", sql_clean, re.IGNORECASE)

        # ORDER BY
        order_match = re.search(r"ORDER\s+BY\s+([\w.]+)\s*(ASC|DESC)?", sql_clean, re.IGNORECASE)

        # JOINがある場合: 2段階クエリで処理
        if join_match:
            return self._exec_join_select(
                sql_clean, params, table, join_match, where_match, order_match, select_cols
            )

        # 通常のSELECT
        # カラム指定
        if select_cols == "*":
            query = self._client.table(table).select("*")
        elif "DISTINCT" in select_cols.upper():
            col = re.search(r"DISTINCT\s+(\w+\.)?(\w+)", select_cols, re.IGNORECASE)
            col_name = col.group(2) if col else "*"
            query = self._client.table(table).select(col_name)
        else:
            # テーブルエイリアスを除去 (rr.finish_position → finish_position)
            cols = []
            for c in select_cols.split(","):
                c = c.strip()
                if "." in c:
                    c = c.split(".")[-1]
                cols.append(c)
            query = self._client.table(table).select(",".join(cols))

        # WHERE 条件を適用
        if where_match:
            query = self._apply_where(query, where_match.group(1), params)

        # ORDER BY
        if order_match:
            col = order_match.group(1)
            if "." in col:
                col = col.split(".")[-1]
            desc = order_match.group(2) and order_match.group(2).upper() == "DESC"
            query = query.order(col, desc=desc)

        try:
            resp = query.execute()
            rows = resp.data if resp.data else []
            return [SupabaseRow(r) for r in rows]
        except Exception as e:
            logger.warning("Supabase query error: %s (SQL: %s)", e, sql_clean[:100])
            return []

    def _exec_join_select(
        self, sql: str, params: tuple,
        main_table: str, join_match, where_match, order_match, select_cols: str
    ) -> list[SupabaseRow]:
        """JOIN付きSELECTの処理 — 2段階クエリで実現"""
        import re

        join_table = join_match.group(1)
        join_alias = join_match.group(2)
        left_alias = join_match.group(3)
        left_col = join_match.group(4)
        right_alias = join_match.group(5)
        right_col = join_match.group(6)

        # WHERE条件からパラメータを分離
        where_conditions = []
        main_conditions = []
        join_conditions = []
        param_idx = 0

        if where_match:
            where_str = where_match.group(1)
            parts = re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                # どのテーブルの条件か判定
                if join_alias + "." in part or join_table + "." in part:
                    count = part.count("?")
                    cond_params = params[param_idx:param_idx + count]
                    param_idx += count
                    join_conditions.append((part, cond_params))
                else:
                    count = part.count("?")
                    cond_params = params[param_idx:param_idx + count]
                    param_idx += count
                    main_conditions.append((part, cond_params))

        # Step 1: JOIN先テーブルから条件に合う行を取得
        if join_conditions or main_conditions:
            # メインテーブルから条件に合うrace_idなどを取得
            q_main = self._client.table(main_table).select("*")
            for cond_str, cond_params in main_conditions:
                q_main = self._apply_single_where(q_main, cond_str, cond_params, "")
            try:
                resp_main = q_main.execute()
                main_rows = {r[left_col]: r for r in (resp_main.data or [])}
            except Exception:
                main_rows = {}

            # JOIN先テーブル
            q_join = self._client.table(join_table).select("*")
            for cond_str, cond_params in join_conditions:
                q_join = self._apply_single_where(q_join, cond_str, cond_params, join_alias)
            if order_match:
                col = order_match.group(1)
                if "." in col:
                    col = col.split(".")[-1]
                desc = order_match.group(2) and order_match.group(2).upper() == "DESC"
                q_join = q_join.order(col, desc=desc)
            try:
                resp_join = q_join.execute()
                join_rows = resp_join.data or []
            except Exception:
                join_rows = []

            # マージ
            results = []
            for jr in join_rows:
                join_key = jr.get(right_col)
                if join_key in main_rows:
                    merged = {**main_rows[join_key], **jr}
                    results.append(SupabaseRow(merged))

            return results

        return []

    def _apply_where(self, query, where_str: str, params: tuple):
        """WHERE句をSupabaseフィルタに変換"""
        import re
        parts = re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)
        param_idx = 0
        for part in parts:
            part = part.strip()
            count = part.count("?")
            part_params = params[param_idx:param_idx + count]
            param_idx += count
            query = self._apply_single_where(query, part, part_params, "")
        return query

    def _apply_single_where(self, query, condition: str, params: tuple, alias: str):
        """1つのWHERE条件をSupabaseフィルタに変換"""
        import re

        condition = condition.strip()

        # テーブルエイリアスを除去
        condition_clean = re.sub(r"\w+\.", "", condition)

        # IS NOT NULL
        match = re.match(r"(\w+)\s+IS\s+NOT\s+NULL", condition_clean, re.IGNORECASE)
        if match:
            return query.not_.is_(match.group(1), "null")

        # IS NULL
        match = re.match(r"(\w+)\s+IS\s+NULL", condition_clean, re.IGNORECASE)
        if match:
            return query.is_(match.group(1), "null")

        # col = ?
        match = re.match(r"(\w+)\s*=\s*\?", condition_clean)
        if match and params:
            return query.eq(match.group(1), params[0])

        # col < ?
        match = re.match(r"(\w+)\s*<\s*\?", condition_clean)
        if match and params:
            return query.lt(match.group(1), params[0])

        # col <= ?
        match = re.match(r"(\w+)\s*<=\s*\?", condition_clean)
        if match and params:
            return query.lte(match.group(1), params[0])

        # col > ?
        match = re.match(r"(\w+)\s*>\s*\?", condition_clean)
        if match and params:
            return query.gt(match.group(1), params[0])

        # col >= ?
        match = re.match(r"(\w+)\s*>=\s*\?", condition_clean)
        if match and params:
            return query.gte(match.group(1), params[0])

        return query

    def _exec_insert(self, sql: str, params: tuple):
        """INSERT文をSupabase upsertに変換"""
        import re

        sql_clean = " ".join(sql.split())

        # INSERT OR REPLACE / INSERT OR IGNORE
        is_upsert = "OR REPLACE" in sql_clean.upper()
        is_ignore = "OR IGNORE" in sql_clean.upper()

        # テーブル名
        table_match = re.search(r"INTO\s+(\w+)", sql_clean, re.IGNORECASE)
        if not table_match:
            return
        table = table_match.group(1)

        # カラム名
        cols_match = re.search(r"\(([^)]+)\)\s*VALUES", sql_clean, re.IGNORECASE)
        if not cols_match:
            return
        columns = [c.strip() for c in cols_match.group(1).split(",")]

        # データ作成（Noneはスキップしない — PostgreSQLはNULLを受け入れる）
        data = {}
        for i, col in enumerate(columns):
            if i < len(params):
                data[col] = params[i]

        try:
            if is_upsert or is_ignore:
                # UPSERT: 主キー/ユニーク制約で判定
                self._client.table(table).upsert(
                    data, on_conflict=self._get_conflict_key(table)
                ).execute()
            else:
                self._client.table(table).insert(data).execute()
        except Exception as e:
            if is_ignore and ("duplicate" in str(e).lower() or "conflict" in str(e).lower()):
                pass  # INSERT OR IGNORE: 重複は無視
            else:
                logger.warning("Supabase insert error on %s: %s", table, e)

    def _get_conflict_key(self, table: str) -> str:
        """テーブルのUPSERT用コンフリクトキーを返す"""
        conflict_keys = {
            "races": "race_id",
            "riders": "rider_id",
            "race_results": "race_id,rider_id",
            "entries": "race_id,rider_id",
            "rider_stats": "rider_id,period",
            "predictions": "race_id,rider_id",
            "prediction_results": "race_id",
        }
        return conflict_keys.get(table, "id")
