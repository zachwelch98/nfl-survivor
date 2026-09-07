"""Stadium geography table: completeness against real data + math sanity."""

from __future__ import annotations

from datetime import date

from survivor.data.stadiums import STADIUMS, timezone_offset_hours, travel_distance_miles


def test_all_schedule_stadium_ids_are_covered():
    from survivor.data.nflverse import load_schedules

    games = load_schedules()
    ids_in_data = set(games["stadium_id"].unique().to_list())
    missing = ids_in_data - set(STADIUMS.keys())
    assert not missing, f"stadium_ids present in schedule but missing from STADIUMS table: {missing}"


def test_distance_is_symmetric_and_zero_for_same_stadium():
    d1 = travel_distance_miles("SEA00", "NYC01")
    d2 = travel_distance_miles("NYC01", "SEA00")
    assert d1 == d2
    assert d1 > 2000  # coast to coast is genuinely far
    assert travel_distance_miles("SEA00", "SEA00") == 0.0


def test_distance_unknown_stadium_returns_none():
    assert travel_distance_miles("SEA00", "NOPE99") is None


def test_timezone_offset_sign_and_symmetry():
    west_to_east = timezone_offset_hours("SEA00", "NYC01", date(2025, 10, 5))
    east_to_west = timezone_offset_hours("NYC01", "SEA00", date(2025, 10, 5))
    assert west_to_east == 3.0
    assert east_to_west == -3.0


def test_timezone_offset_same_stadium_is_zero():
    assert timezone_offset_hours("NYC01", "NYC01", date(2025, 10, 5)) == 0.0
