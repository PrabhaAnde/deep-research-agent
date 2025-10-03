#!/usr/bin/env python3
"""
Date Utilities for Research Agents
Handles date parsing, calculation, and formatting
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import re


class DateUtils:
    """Utilities for date handling in research agents"""

    @staticmethod
    def parse_reference_date(date_str: Optional[str] = None) -> datetime:
        """
        Parse reference date from string or return current date

        Args:
            date_str: Date string in various formats:
                - "2025-09-29"
                - "September 29, 2025"
                - "09/29/2025"
                - "today" or None (returns current date)

        Returns:
            datetime object

        Examples:
            >>> DateUtils.parse_reference_date("2025-09-29")
            datetime(2025, 9, 29, 0, 0)

            >>> DateUtils.parse_reference_date("today")
            datetime(2025, 9, 30, 0, 0)  # Current date
        """
        if not date_str or date_str.lower() == "today":
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Try various formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y/%m/%d"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Could not parse date: {date_str}. Supported formats: YYYY-MM-DD, MM/DD/YYYY, 'Month DD, YYYY'")

    @staticmethod
    def extract_date_from_query(query: str) -> Optional[datetime]:
        """
        Extract date from natural language query

        Args:
            query: User query that may contain dates

        Returns:
            datetime object if date found, None otherwise

        Examples:
            >>> DateUtils.extract_date_from_query("from today(9/29/2025)")
            datetime(2025, 9, 29, 0, 0)

            >>> DateUtils.extract_date_from_query("as of September 29, 2025")
            datetime(2025, 9, 29, 0, 0)
        """
        # Pattern: today(DATE)
        match = re.search(r'today\s*\(([^\)]+)\)', query)
        if match:
            try:
                return DateUtils.parse_reference_date(match.group(1))
            except:
                pass

        # Pattern: as of DATE
        match = re.search(r'as of\s+([A-Za-z]+ \d+,? \d{4})', query)
        if match:
            try:
                return DateUtils.parse_reference_date(match.group(1))
            except:
                pass

        # Pattern: on DATE
        match = re.search(r'on\s+(\d{4}-\d{2}-\d{2})', query)
        if match:
            try:
                return DateUtils.parse_reference_date(match.group(1))
            except:
                pass

        return None

    @staticmethod
    def calculate_date_range(
        reference_date: datetime,
        range_type: str = "6months"
    ) -> Tuple[datetime, datetime]:
        """
        Calculate date range relative to reference date

        Args:
            reference_date: The reference "today" date
            range_type: One of:
                - "1week" or "week": Past 7 days
                - "2weeks": Past 14 days
                - "1month" or "month": Past 30 days
                - "3months": Past 90 days
                - "6months": Past 180 days
                - "1year" or "year": Past 365 days

        Returns:
            Tuple of (start_date, end_date)

        Examples:
            >>> ref = datetime(2025, 9, 29)
            >>> DateUtils.calculate_date_range(ref, "1week")
            (datetime(2025, 9, 22, 0, 0), datetime(2025, 9, 29, 0, 0))

            >>> DateUtils.calculate_date_range(ref, "6months")
            (datetime(2025, 3, 31, 0, 0), datetime(2025, 9, 29, 0, 0))
        """
        range_map = {
            "1week": 7,
            "week": 7,
            "2weeks": 14,
            "1month": 30,
            "month": 30,
            "3months": 90,
            "6months": 180,
            "1year": 365,
            "year": 365
        }

        days = range_map.get(range_type.lower())
        if not days:
            raise ValueError(f"Unknown range_type: {range_type}. Supported: {list(range_map.keys())}")

        start_date = reference_date - timedelta(days=days)
        return (start_date, reference_date)

    @staticmethod
    def extract_date_ranges_from_query(query: str, reference_date: datetime) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Extract multiple date ranges from query

        Args:
            query: User query
            reference_date: Reference "today" date

        Returns:
            Dictionary mapping range names to (start_date, end_date) tuples

        Examples:
            >>> ref = datetime(2025, 9, 29)
            >>> DateUtils.extract_date_ranges_from_query("past 6 months... past week", ref)
            {'6months': (datetime(2025, 3, 31), datetime(2025, 9, 29)),
             'week': (datetime(2025, 9, 22), datetime(2025, 9, 29))}
        """
        ranges = {}

        # Patterns to look for
        patterns = [
            (r'past\s+(\d+)\s+months?', lambda m: f"{m.group(1)}months"),
            (r'past\s+(\d+)\s+weeks?', lambda m: f"{m.group(1)}weeks"),
            (r'past\s+(\d+)\s+years?', lambda m: f"{m.group(1)}year"),
            (r'past\s+6\s+months?', lambda m: "6months"),
            (r'past\s+week', lambda m: "week"),
            (r'past\s+month', lambda m: "month"),
            (r'past\s+year', lambda m: "year"),
            (r'last\s+week', lambda m: "week"),
            (r'last\s+month', lambda m: "month"),
        ]

        for pattern, range_getter in patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                range_key = range_getter(match)

                # Parse custom number of months/weeks
                if re.match(r'\d+months?', range_key):
                    num = int(re.search(r'\d+', range_key).group())
                    days = num * 30
                    start = reference_date - timedelta(days=days)
                    ranges[range_key] = (start, reference_date)
                elif re.match(r'\d+weeks?', range_key):
                    num = int(re.search(r'\d+', range_key).group())
                    days = num * 7
                    start = reference_date - timedelta(days=days)
                    ranges[range_key] = (start, reference_date)
                else:
                    try:
                        ranges[range_key] = DateUtils.calculate_date_range(reference_date, range_key)
                    except ValueError:
                        pass

        return ranges

    @staticmethod
    def format_date_for_display(dt: datetime, format_type: str = "readable") -> str:
        """
        Format date for display

        Args:
            dt: datetime object
            format_type: One of:
                - "readable": "September 29, 2025"
                - "short": "Sep 29, 2025"
                - "iso": "2025-09-29"
                - "us": "09/29/2025"

        Returns:
            Formatted date string
        """
        formats = {
            "readable": "%B %d, %Y",
            "short": "%b %d, %Y",
            "iso": "%Y-%m-%d",
            "us": "%m/%d/%Y"
        }

        fmt = formats.get(format_type, formats["readable"])
        return dt.strftime(fmt)

    @staticmethod
    def format_date_range_for_display(start: datetime, end: datetime) -> str:
        """
        Format date range for display

        Examples:
            >>> start = datetime(2025, 3, 31)
            >>> end = datetime(2025, 9, 29)
            >>> DateUtils.format_date_range_for_display(start, end)
            "March 31, 2025 to September 29, 2025"
        """
        return f"{DateUtils.format_date_for_display(start)} to {DateUtils.format_date_for_display(end)}"

    @staticmethod
    def is_date_in_range(date_str: str, start_date: datetime, end_date: datetime) -> bool:
        """
        Check if a relative date string falls within range

        Args:
            date_str: Relative date like "3 days ago", "1 week ago", "2 months ago"
            start_date: Range start
            end_date: Range end (usually "today")

        Returns:
            True if date is within range

        Examples:
            >>> end = datetime(2025, 9, 29)
            >>> start = end - timedelta(days=7)
            >>> DateUtils.is_date_in_range("3 days ago", start, end)
            True
            >>> DateUtils.is_date_in_range("2 months ago", start, end)
            False
        """
        # Parse relative dates
        patterns = [
            (r'(\d+)\s+days?\s+ago', lambda m: timedelta(days=int(m.group(1)))),
            (r'(\d+)\s+weeks?\s+ago', lambda m: timedelta(weeks=int(m.group(1)))),
            (r'(\d+)\s+months?\s+ago', lambda m: timedelta(days=int(m.group(1)) * 30)),
            (r'(\d+)\s+years?\s+ago', lambda m: timedelta(days=int(m.group(1)) * 365)),
        ]

        for pattern, delta_getter in patterns:
            match = re.search(pattern, date_str.lower())
            if match:
                delta = delta_getter(match)
                article_date = end_date - delta
                return start_date <= article_date <= end_date

        # If we can't parse, assume it's in range (benefit of doubt)
        return True

    @staticmethod
    def calculate_future_date(reference_date: datetime, days_ahead: int) -> datetime:
        """
        Calculate future date

        Args:
            reference_date: Starting date
            days_ahead: Number of days in the future

        Returns:
            Future datetime

        Examples:
            >>> ref = datetime(2025, 9, 29)
            >>> DateUtils.calculate_future_date(ref, 30)
            datetime(2025, 10, 29, 0, 0)
        """
        return reference_date + timedelta(days=days_ahead)


# Convenience functions (can be used as tools)

def get_date_range(reference_date_str: str, range_type: str) -> Dict:
    """
    Tool-friendly function to get date range

    Args:
        reference_date_str: Reference date as string or "today"
        range_type: Range type (e.g., "6months", "week")

    Returns:
        Dictionary with start_date, end_date, and formatted strings

    Example:
        >>> get_date_range("2025-09-29", "6months")
        {
            'start_date': '2025-03-31',
            'end_date': '2025-09-29',
            'formatted': 'March 31, 2025 to September 29, 2025',
            'days': 180
        }
    """
    ref_date = DateUtils.parse_reference_date(reference_date_str)
    start, end = DateUtils.calculate_date_range(ref_date, range_type)

    return {
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date': end.strftime('%Y-%m-%d'),
        'start_date_readable': DateUtils.format_date_for_display(start),
        'end_date_readable': DateUtils.format_date_for_display(end),
        'formatted': DateUtils.format_date_range_for_display(start, end),
        'days': (end - start).days
    }


def parse_query_dates(query: str) -> Dict:
    """
    Tool-friendly function to extract dates from query

    Args:
        query: User query

    Returns:
        Dictionary with reference_date and date_ranges

    Example:
        >>> parse_query_dates("past 6 months... past week from today(9/29/2025)")
        {
            'reference_date': '2025-09-29',
            'reference_date_readable': 'September 29, 2025',
            'date_ranges': {
                '6months': {...},
                'week': {...}
            }
        }
    """
    # Extract reference date
    ref_date = DateUtils.extract_date_from_query(query)
    if not ref_date:
        ref_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Extract ranges
    ranges_dict = DateUtils.extract_date_ranges_from_query(query, ref_date)

    # Format for output
    formatted_ranges = {}
    for name, (start, end) in ranges_dict.items():
        formatted_ranges[name] = {
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'formatted': DateUtils.format_date_range_for_display(start, end),
            'days': (end - start).days
        }

    return {
        'reference_date': ref_date.strftime('%Y-%m-%d'),
        'reference_date_readable': DateUtils.format_date_for_display(ref_date),
        'date_ranges': formatted_ranges
    }


if __name__ == "__main__":
    # Test the utilities
    print("Date Utilities Test")
    print("=" * 60)

    # Test 1: Parse reference date
    ref = DateUtils.parse_reference_date("2025-09-29")
    print(f"\n1. Parsed reference date: {ref}")

    # Test 2: Calculate ranges
    ranges = ["week", "month", "6months"]
    for r in ranges:
        start, end = DateUtils.calculate_date_range(ref, r)
        print(f"\n2. Range '{r}':")
        print(f"   {DateUtils.format_date_range_for_display(start, end)}")
        print(f"   Days: {(end - start).days}")

    # Test 3: Extract from query
    query = "past 6 months... past week from today(9/29/2025)"
    print(f"\n3. Query: {query}")

    extracted_ref = DateUtils.extract_date_from_query(query)
    print(f"   Reference date: {extracted_ref}")

    extracted_ranges = DateUtils.extract_date_ranges_from_query(query, ref)
    print(f"   Ranges found: {list(extracted_ranges.keys())}")
    for name, (start, end) in extracted_ranges.items():
        print(f"     {name}: {DateUtils.format_date_range_for_display(start, end)}")

    # Test 4: Tool functions
    print("\n4. Tool function: get_date_range")
    result = get_date_range("2025-09-29", "6months")
    print(f"   {result}")

    print("\n5. Tool function: parse_query_dates")
    result = parse_query_dates(query)
    print(f"   Reference: {result['reference_date_readable']}")
    print(f"   Ranges: {list(result['date_ranges'].keys())}")