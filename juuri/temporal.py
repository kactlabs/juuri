"""Dates for temporal facts. Everything is stored as ISO "YYYY-MM-DD" strings."""

import calendar
import re
from datetime import date, datetime
from typing import Iterable, Optional, Union

DateLike = Union[str, int, date, None]

_PARTIAL = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
_FULL = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date(value: DateLike, end: bool = False) -> Optional[str]:
    """Normalise a date to "YYYY-MM-DD". None and "" give None.

    Accepts a date or datetime, an int year, or a string "YYYY", "YYYY-MM",
    "YYYY-MM-DD" (a time after "T" or a space is ignored). A partial date
    expands to the start of the period, or to its end if end=True, so
    valid_to="2016" means "through 2016-12-31".
    Raises ValueError for anything else.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"invalid date {value!r}")
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "T").split("T")[0]
    match = _PARTIAL.match(text)
    if not match:
        raise ValueError(f"invalid date {value!r}: use YYYY, YYYY-MM or YYYY-MM-DD")
    year, month, day = match.groups()
    y = int(year)
    if month is None:
        m, d = (12, 31) if end else (1, 1)
    else:
        m = int(month)
        if day is not None:
            d = int(day)
        else:
            d = calendar.monthrange(y, m)[1] if end else 1
    return date(y, m, d).isoformat()


def parse_as_of(value: Union[str, date]) -> str:
    """Normalise an "as of" date. A string must be a full "YYYY-MM-DD".

    A partial date such as "2012" is rejected because it does not name one day.
    """
    if isinstance(value, str) and not _FULL.match(value.strip().replace(" ", "T").split("T")[0]):
        raise ValueError(f"as_of must be a full date YYYY-MM-DD, got {value!r}")
    result = parse_date(value)
    if result is None:
        raise ValueError("as_of must not be empty")
    return result


def covers(records: Iterable, day: str, include_undated: bool = True) -> bool:
    """True if the provenance records say the fact holds on `day` (ISO string).

    Dated records decide when there are any: the fact holds if one of their
    periods contains the day (both ends inclusive, a missing end is open).
    Undated records only count when no record is dated, and then the answer
    is include_undated.
    """
    dated = [r for r in records if r.valid_from or r.valid_to]
    if not dated:
        return include_undated
    return any(
        (r.valid_from is None or r.valid_from <= day) and (r.valid_to is None or day <= r.valid_to)
        for r in dated
    )
