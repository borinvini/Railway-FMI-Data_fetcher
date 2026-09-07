# -*- coding: utf-8 -*-
import pandas as pd
import requests
from defusedxml.ElementTree import fromstring

from config.const import FMI_EMS, FMI_WEATHER_NETWORKS, FMI_WFS_BASE

# INSPIRE Environmental Facility schema namespaces.
EF_NAMESPACES = {
    "ef": "http://inspire.ec.europa.eu/schemas/ef/4.0",
    "gml": "http://www.opengis.net/gml/3.2",
}

# Network membership is carried on the xlink:title attribute of ef:belongsTo.
XLINK_TITLE = "{http://www.w3.org/1999/xlink}title"

REGISTRY_COLUMNS = [
    "fmisid", "station_name", "latitude", "longitude",
    "networks", "is_weather_station",
]


def parse_ef_stations(xml_bytes):
    """
    Parses an fmi::ef::stations WFS response into a station registry DataFrame.

    fmiopendata cannot decode this schema (it raises NotImplementedError), so the
    response is parsed directly. Facilities without a gml:pos are skipped, since a
    station with no coordinates cannot participate in distance matching.

    Args:
        xml_bytes (bytes): Raw WFS response body.

    Returns:
        pd.DataFrame: One row per locatable facility, columns REGISTRY_COLUMNS.
    """
    root = fromstring(xml_bytes)
    rows = []
    skipped = 0

    for facility in root.findall(".//ef:EnvironmentalMonitoringFacility", EF_NAMESPACES):
        position = facility.findtext(".//gml:pos", namespaces=EF_NAMESPACES)
        if not position:
            skipped += 1
            continue

        coordinates = position.split()
        if len(coordinates) < 2:
            skipped += 1
            continue

        # srsName is EPSG:4258 with axisLabels "Lat Long", so latitude comes first.
        networks = sorted({
            membership.get(XLINK_TITLE)
            for membership in facility.findall("ef:belongsTo", EF_NAMESPACES)
            if membership.get(XLINK_TITLE)
        })

        rows.append({
            "fmisid": int(facility.findtext("gml:identifier", namespaces=EF_NAMESPACES)),
            "station_name": facility.findtext("ef:name", namespaces=EF_NAMESPACES),
            "latitude": float(coordinates[0]),
            "longitude": float(coordinates[1]),
            "networks": "|".join(networks),
            "is_weather_station": any(n in FMI_WEATHER_NETWORKS for n in networks),
        })

    if skipped:
        print(f"⚠️ Skipped {skipped} EF facilities without coordinates.")

    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


class FMIStationRegistry:
    """
    Fetches the FMI environmental facility catalogue (fmi::ef::stations).

    The catalogue is an enrichment and validation source, not the station list.
    Station discovery stays with the observation feed; this supplies authoritative
    coordinates, station typing, and the ability to tell an absent station from a
    silent one.
    """

    def __init__(self, base_url=FMI_WFS_BASE, stored_query=FMI_EMS, timeout=180):
        self.base_url = base_url
        self.stored_query = stored_query
        self.timeout = timeout

    def fetch_registry(self):
        """
        Fetches and parses the full EF station catalogue.

        A registry outage must not abort a multi-hour observation fetch, so failures
        are reported and an empty DataFrame is returned rather than raised.

        Returns:
            pd.DataFrame: Station registry, or an empty frame with REGISTRY_COLUMNS.
        """
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": self.stored_query,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            registry = parse_ef_stations(response.content)
        except Exception as error:
            print(f"⚠️ Could not fetch the EF station registry: {error}")
            print("   Continuing with observation-derived station metadata only.")
            return pd.DataFrame(columns=REGISTRY_COLUMNS)

        weather_count = int(registry["is_weather_station"].sum())
        print(f"✅ EF registry: {len(registry)} facilities, {weather_count} weather stations.")
        return registry
