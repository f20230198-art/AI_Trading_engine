"""
DOOM ENGINE — SQL Guard

Every SQL statement the Analyst agent wants to run passes through two
independent layers. The LLM that wrote the SQL is never trusted.

Layer 1 — static, before execution (sqlglot AST, not regex):
  - exactly one statement (stacked queries rejected)
  - the root must be a SELECT / UNION / INTERSECT / EXCEPT
  - no DDL, DML, PRAGMA, ATTACH, transactions or SELECT ... INTO anywhere in the tree
  - every table must be in the allow-list (or a CTE defined in the same query);
    schema-qualified names (main.x, other_db.x) are rejected
  - every column must exist in a referenced table, or be an alias defined in the query
  - every function must be in the allow-list (blocks load_extension, readfile, ...)
  - a LIMIT is forced on the outermost query
  The query that runs is regenerated from the validated AST with comments
  stripped, so what was checked is exactly what executes.

Layer 2 — at execution, inside SQLite:
  - read-only connection (see core/analytics_db.connect_readonly)
  - an authorizer callback that SQLite consults for every table/column read and
    every function call, denying anything outside the allow-list
  - a progress handler that aborts the query after a wall-clock timeout
  - result caps on row count and per-cell size
"""
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp

from core.analytics_db import SCHEMA_DOCS

MAX_SQL_CHARS = 4000
MAX_ROWS = 500            # hard LIMIT forced on every query
QUERY_TIMEOUT_S = 2.0
MAX_CELL_CHARS = 300      # long free-text values are truncated before leaving the DB layer

ALLOWED_COLUMNS: Dict[str, Set[str]] = {
    name: set(doc["columns"]) for name, doc in SCHEMA_DOCS.items()
}

# Functions as sqlglot types them (it normalizes many SQLite functions to typed nodes).
_ALLOWED_FUNC_NODES = tuple(getattr(exp, name) for name in (
    "Count", "Sum", "Avg", "Min", "Max", "Round", "Abs", "Coalesce", "Nullif", "Substring", "Cast", "TryCast", "If", "Case", "Lower", "Upper", "Length", "Trim", "Replace", "StrPosition", "TimeToStr", "Date", "TsOrDsToTimestamp", "Sqrt", "GroupConcat", "RowNumber", "Rank", "DenseRank", "Lag", "Lead", "FirstValue", "LastValue", "NthValue", "Ntile", "PercentRank", "CumeDist", "Stddev", "StddevSamp", "StddevPop", "Variance", "Greatest", "Least",
) if hasattr(exp, name))
# Functions sqlglot leaves as Anonymous, matched by lowercase name.
_ALLOWED_ANON_FUNCS = {"julianday", "datetime", "time", "total", "printf", "iif", "strftime", "date"}

# Exact SQLite function names the authorizer (layer 2) will permit.
SQLITE_ALLOWED_FUNCS = {
    "count", "sum", "avg", "min", "max", "round", "abs", "coalesce", "ifnull", "nullif",
    "substr", "substring", "lower", "upper", "length", "trim", "ltrim", "rtrim", "replace",
    "instr", "strftime", "date", "datetime", "time", "julianday", "sqrt", "group_concat",
    "row_number", "rank", "dense_rank", "lag", "lead", "first_value", "last_value",
    "nth_value", "ntile", "percent_rank", "cume_dist", "total", "printf", "format", "iif",
    "like", "glob", "cast",
}

_FORBIDDEN_NODES = tuple(
    getattr(exp, name) for name in (
        "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "TruncateTable",
        "Command", "Pragma", "Attach", "Detach", "Transaction", "Commit", "Rollback",
        "Into", "Set", "Use", "LoadData", "Copy",
    ) if hasattr(exp, name)
)


class SQLGuardError(Exception):
    """A query was rejected. `code` is stable and machine-readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class QueryResult:
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    hit_row_cap: bool
    elapsed_ms: float
    cells_truncated: int = 0


@dataclass
class _Scope:
    ctes: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)
    cte_columns: Set[str] = field(default_factory=set)


def validate_sql(sql: str, max_rows: int = MAX_ROWS) -> str:
    """
    Layer 1. Returns the safe, regenerated SQL to execute, or raises SQLGuardError.
    """
    if not sql or not sql.strip():
        raise SQLGuardError("EMPTY", "No SQL was produced.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLGuardError("TOO_LONG", f"SQL longer than {MAX_SQL_CHARS} characters.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLGuardError("PARSE_ERROR", f"SQL could not be parsed: {str(e)[:200]}")
    if len(statements) != 1:
        raise SQLGuardError("MULTIPLE_STATEMENTS", "Exactly one SQL statement is allowed.")

    root = statements[0]
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        raise SQLGuardError("NOT_SELECT", f"Only SELECT queries are allowed, got {root.key.upper()}.")

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLGuardError("FORBIDDEN_STATEMENT", f"{node.key.upper()} is not allowed.")

    scope = _Scope()
    for cte in root.find_all(exp.CTE):
        scope.ctes.add(cte.alias_or_name.lower())
        for col in (cte.args.get("alias") and cte.args["alias"].columns) or []:
            scope.cte_columns.add(col.name.lower())
    for alias in root.find_all(exp.Alias):
        scope.aliases.add(alias.alias.lower())
    for table_alias in root.find_all(exp.TableAlias):
        scope.aliases.add(table_alias.name.lower())

    referenced = _check_tables(root, scope)
    _check_columns(root, scope, referenced)
    _check_functions(root)

    _force_limit(root, max_rows)
    return root.sql(dialect="sqlite", comments=False)


def _check_tables(root: exp.Expression, scope: _Scope) -> Set[str]:
    referenced = set()
    for table in root.find_all(exp.Table):
        name = table.name.lower()
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLGuardError("DISALLOWED_TABLE", f"Schema-qualified table '{table.sql()}' is not allowed.")
        if not name:
            raise SQLGuardError("DISALLOWED_TABLE", "Table-valued expressions are not allowed.")
        if name in scope.ctes:
            continue
        if name not in ALLOWED_COLUMNS:
            raise SQLGuardError("DISALLOWED_TABLE", f"Table '{name}' is not in the allow-list.")
        referenced.add(name)
    return referenced


def _check_columns(root: exp.Expression, scope: _Scope, referenced: Set[str]):
    visible = set().union(*(ALLOWED_COLUMNS[t] for t in referenced)) if referenced else set()
    for column in root.find_all(exp.Column):
        name = column.name.lower()
        if not name or name == "*":
            continue
        if name in visible or name in scope.aliases or name in scope.cte_columns:
            continue
        raise SQLGuardError("DISALLOWED_COLUMN", f"Column '{column.sql()}' does not exist in the allowed tables.")


# sqlglot models operators (AND, IN, BETWEEN, =, LIKE, ...) as Func subclasses too;
# they are plain SQL syntax, not function calls, so the allow-list skips them.
_OPERATOR_NODES = (exp.Binary, exp.Connector, exp.Predicate, exp.Unary)


def _check_functions(root: exp.Expression):
    for func in root.find_all(exp.Func):
        if isinstance(func, _OPERATOR_NODES):
            continue
        if isinstance(func, exp.Anonymous):
            if func.name.lower() not in _ALLOWED_ANON_FUNCS:
                raise SQLGuardError("DISALLOWED_FUNCTION", f"Function '{func.name}' is not allowed.")
        elif not isinstance(func, _ALLOWED_FUNC_NODES) and not isinstance(func, exp.AggFunc) \
                or isinstance(func, (exp.Rand,)):
            raise SQLGuardError("DISALLOWED_FUNCTION", f"Function '{func.sql_name()}' is not allowed.")


def _force_limit(root: exp.Expression, max_rows: int):
    limit = root.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.this) if limit.expression else None
        except (TypeError, ValueError, AttributeError):
            current = None
    if current is None or current > max_rows:
        # One extra row lets execute_guarded tell "exactly max_rows" from "truncated".
        root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows + 1)))


# ---------------------------------------------------------------------------
# Layer 2 — execution sandbox
# ---------------------------------------------------------------------------
def _authorizer(action, arg1, arg2, db_name, trigger_or_view):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # arg1 = table, arg2 = column. Views are expanded, so a read through an
        # allowed view reports the underlying base table, which is also allowed.
        table = (arg1 or "").lower()
        if table in ALLOWED_COLUMNS and (not arg2 or arg2.lower() in ALLOWED_COLUMNS[table]):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in SQLITE_ALLOWED_FUNCS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY  # PRAGMA, ATTACH, writes, recursive CTEs, everything else


def execute_guarded(conn: sqlite3.Connection, sql: str,
                    timeout_s: float = QUERY_TIMEOUT_S, max_rows: int = MAX_ROWS) -> QueryResult:
    """Validate (layer 1) and run under the authorizer and timeout (layer 2)."""
    safe_sql = validate_sql(sql, max_rows=max_rows)
    deadline = time.monotonic() + timeout_s
    start = time.perf_counter()

    conn.set_authorizer(_authorizer)
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        cursor = conn.execute(safe_sql)
        raw_rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except sqlite3.DatabaseError as e:
        message = str(e)
        if "interrupted" in message:
            raise SQLGuardError("TIMEOUT", f"Query exceeded {timeout_s:.1f}s and was aborted.")
        if "not authorized" in message or "prohibited" in message:
            raise SQLGuardError("NOT_AUTHORIZED", "The database refused this query.")
        raise SQLGuardError("EXECUTION_ERROR", message[:200])
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)

    truncated_cells = 0
    rows = []
    for raw in raw_rows[:max_rows]:
        row = []
        for value in raw:
            if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
                value = value[:MAX_CELL_CHARS] + "…"
                truncated_cells += 1
            elif isinstance(value, bytes):
                value = f"<{len(value)} bytes>"
                truncated_cells += 1
            row.append(value)
        rows.append(row)

    return QueryResult(
        sql=safe_sql,
        columns=columns,
        rows=rows,
        hit_row_cap=len(raw_rows) > max_rows,
        elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
        cells_truncated=truncated_cells,
    )
