"""Static stadium geography table: lat/lon/timezone by nflverse `stadium_id`.

nflverse's schedule data has no coordinates, so this hand-built table (67
stadium_ids covering every venue that has hosted a game 1999-present,
including international sites) is what makes travel-distance and
timezone-crossing features possible. It is deliberately just geography —
resolving "which stadium is team X's home base in season S" (teams relocate;
stadiums get renovated and renamed) is a job for features/situational.py,
which has the schedule available to look that up empirically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class StadiumInfo:
    stadium_id: str
    name: str
    lat: float
    lon: float
    tz: str  # IANA timezone name


# lat/lon in decimal degrees; tz is the IANA zone the stadium physically sits in.
STADIUMS: dict[str, StadiumInfo] = {
    s.stadium_id: s
    for s in [
        StadiumInfo("ATL00", "Georgia Dome", 33.7573, -84.4006, "America/New_York"),
        StadiumInfo("ATL97", "Mercedes-Benz Stadium", 33.7554, -84.4008, "America/New_York"),
        StadiumInfo("BAL00", "M&T Bank Stadium", 39.2780, -76.6227, "America/New_York"),
        StadiumInfo("BOS00", "Gillette Stadium", 42.0909, -71.2643, "America/New_York"),
        StadiumInfo("BOS99", "Foxboro Stadium", 42.0654, -71.2536, "America/New_York"),
        StadiumInfo("BRG00", "Tiger Stadium (LSU)", 30.4118, -91.1837, "America/Chicago"),
        StadiumInfo("BUF00", "Highmark Stadium", 42.7738, -78.7870, "America/New_York"),
        StadiumInfo("BUF01", "Rogers Centre", 43.6414, -79.3894, "America/Toronto"),
        StadiumInfo("CAR00", "Bank of America Stadium", 35.2258, -80.8528, "America/New_York"),
        StadiumInfo("CHI98", "Soldier Field", 41.8623, -87.6167, "America/Chicago"),
        StadiumInfo("CHI99", "Memorial Stadium (Champaign)", 40.0994, -88.2359, "America/Chicago"),
        StadiumInfo("CIN00", "Paycor Stadium", 39.0954, -84.5160, "America/New_York"),
        StadiumInfo("CIN99", "Cinergy Field", 39.0975, -84.5147, "America/New_York"),
        StadiumInfo("CLE00", "Huntington Bank Field", 41.5061, -81.6995, "America/New_York"),
        StadiumInfo("DAL00", "AT&T Stadium", 32.7473, -97.0945, "America/Chicago"),
        StadiumInfo("DAL99", "Texas Stadium", 32.8408, -96.9111, "America/Chicago"),
        StadiumInfo("DEN00", "Empower Field at Mile High", 39.7439, -105.0201, "America/Denver"),
        StadiumInfo("DEN99", "Mile High Stadium", 39.7439, -105.0201, "America/Denver"),
        StadiumInfo("DET00", "Ford Field", 42.3400, -83.0456, "America/Detroit"),
        StadiumInfo("DET99", "Pontiac Silverdome", 42.6489, -83.2586, "America/Detroit"),
        StadiumInfo("FRA00", "Deutsche Bank Park", 50.0686, 8.6455, "Europe/Berlin"),
        StadiumInfo("GER00", "Allianz Arena", 48.2188, 11.6247, "Europe/Berlin"),
        StadiumInfo("GNB00", "Lambeau Field", 44.5013, -88.0622, "America/Chicago"),
        StadiumInfo("HOU00", "NRG Stadium", 29.6847, -95.4107, "America/Chicago"),
        StadiumInfo("IND00", "Lucas Oil Stadium", 39.7601, -86.1639, "America/Indiana/Indianapolis"),
        StadiumInfo("IND99", "RCA Dome", 39.7684, -86.1581, "America/Indiana/Indianapolis"),
        StadiumInfo("JAX00", "EverBank Stadium", 30.3239, -81.6373, "America/New_York"),
        StadiumInfo("KAN00", "GEHA Field at Arrowhead Stadium", 39.0489, -94.4839, "America/Chicago"),
        StadiumInfo("LAX01", "SoFi Stadium", 33.9535, -118.3392, "America/Los_Angeles"),
        StadiumInfo("LAX97", "StubHub Center", 33.8644, -118.2611, "America/Los_Angeles"),
        StadiumInfo("LAX99", "Los Angeles Memorial Coliseum", 34.0141, -118.2879, "America/Los_Angeles"),
        StadiumInfo("LON00", "Wembley Stadium", 51.5560, -0.2795, "Europe/London"),
        StadiumInfo("LON01", "Twickenham Stadium", 51.4562, -0.3416, "Europe/London"),
        StadiumInfo("LON02", "Tottenham Hotspur Stadium", 51.6043, -0.0664, "Europe/London"),
        StadiumInfo("MAD01", "Santiago Bernabeu", 40.4531, -3.6883, "Europe/Madrid"),
        StadiumInfo("MEL00", "Melbourne Cricket Ground", -37.8199, 144.9834, "Australia/Melbourne"),
        StadiumInfo("MEX00", "Estadio Azteca / Banorte", 19.3029, -99.1505, "America/Mexico_City"),
        StadiumInfo("MIA00", "Hard Rock Stadium", 25.9580, -80.2389, "America/New_York"),
        StadiumInfo("MIN00", "Mall of America Field", 44.9738, -93.2581, "America/Chicago"),
        StadiumInfo("MIN01", "U.S. Bank Stadium", 44.9735, -93.2575, "America/Chicago"),
        StadiumInfo("MIN98", "TCF Bank Stadium", 44.9760, -93.2245, "America/Chicago"),
        StadiumInfo("MUN01", "FC Bayern Munich Stadium", 48.2188, 11.6247, "Europe/Berlin"),
        StadiumInfo("NAS00", "Nissan Stadium", 36.1665, -86.7713, "America/Chicago"),
        StadiumInfo("NOR00", "Caesars Superdome", 29.9511, -90.0812, "America/Chicago"),
        StadiumInfo("NYC00", "Giants Stadium", 40.8135, -74.0745, "America/New_York"),
        StadiumInfo("NYC01", "MetLife Stadium", 40.8128, -74.0742, "America/New_York"),
        StadiumInfo("OAK00", "Oakland-Alameda County Coliseum", 37.7516, -122.2005, "America/Los_Angeles"),
        StadiumInfo("PAR00", "Stade de France", 48.9244, 2.3601, "Europe/Paris"),
        StadiumInfo("PHI00", "Lincoln Financial Field", 39.9008, -75.1675, "America/New_York"),
        StadiumInfo("PHI99", "Veterans Stadium", 39.9057, -75.1681, "America/New_York"),
        StadiumInfo("PHO00", "State Farm Stadium", 33.5276, -112.2626, "America/Phoenix"),
        StadiumInfo("PHO99", "Sun Devil Stadium", 33.4260, -111.9327, "America/Phoenix"),
        StadiumInfo("PIT00", "Acrisure Stadium", 40.4468, -80.0158, "America/New_York"),
        StadiumInfo("PIT99", "Three Rivers Stadium", 40.4463, -80.0106, "America/New_York"),
        StadiumInfo("RIO00", "Maracana Stadium", -22.9121, -43.2302, "America/Sao_Paulo"),
        StadiumInfo("SAN00", "Alamodome", 29.4169, -98.4769, "America/Chicago"),
        StadiumInfo("SAO00", "Arena Corinthians", -23.5453, -46.4742, "America/Sao_Paulo"),
        StadiumInfo("SDG00", "Qualcomm Stadium", 32.7831, -117.1195, "America/Los_Angeles"),
        StadiumInfo("SEA00", "Lumen Field", 47.5952, -122.3316, "America/Los_Angeles"),
        StadiumInfo("SEA98", "Seattle Kingdome", 47.5952, -122.3316, "America/Los_Angeles"),
        StadiumInfo("SEA99", "Husky Stadium", 47.6503, -122.3017, "America/Los_Angeles"),
        StadiumInfo("SFO00", "Candlestick Park", 37.7135, -122.3860, "America/Los_Angeles"),
        StadiumInfo("SFO01", "Levi's Stadium", 37.4033, -121.9694, "America/Los_Angeles"),
        StadiumInfo("STL00", "Edward Jones Dome", 38.6329, -90.1886, "America/Chicago"),
        StadiumInfo("TAM00", "Raymond James Stadium", 27.9759, -82.5033, "America/New_York"),
        StadiumInfo("VEG00", "Allegiant Stadium", 36.0909, -115.1833, "America/Los_Angeles"),
        StadiumInfo("WAS00", "Northwest Stadium", 38.9077, -76.8645, "America/New_York"),
    ]
}

_EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def travel_distance_miles(from_id: str, to_id: str) -> float | None:
    """Great-circle distance between two stadium_ids, or None if either is unknown."""
    a, b = STADIUMS.get(from_id), STADIUMS.get(to_id)
    if a is None or b is None:
        return None
    return haversine_miles(a.lat, a.lon, b.lat, b.lon)


def timezone_offset_hours(from_id: str, to_id: str, on: date) -> float | None:
    """UTC-offset difference (to minus from) in hours on a given date.

    Positive means the destination clock is ahead of the origin's (an
    eastward trip — the classically harder direction for a 1pm ET kickoff).
    `on` matters because of DST, so pass the actual game date.
    """
    a, b = STADIUMS.get(from_id), STADIUMS.get(to_id)
    if a is None or b is None:
        return None
    # Compare UTC offsets at local noon on the given date to avoid DST-transition edge cases.
    ref = date(on.year, on.month, on.day)
    off_a = ZoneInfo(a.tz).utcoffset(_noon(ref))
    off_b = ZoneInfo(b.tz).utcoffset(_noon(ref))
    if off_a is None or off_b is None:
        return None
    return (off_b - off_a).total_seconds() / 3600.0


def _noon(d: date):
    from datetime import datetime

    return datetime(d.year, d.month, d.day, 12, 0, 0)
