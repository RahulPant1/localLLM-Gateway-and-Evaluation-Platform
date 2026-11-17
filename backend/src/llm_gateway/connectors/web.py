"""Three free public web APIs, as fixed tools: weather, exchange rates and
Wikipedia summaries.

Fixed tools, not "fetch any URL": a model reading untrusted text (an email, a
file) cannot be talked into sending data to an address of an attacker's
choosing. Only the city, currency codes or topic leave the machine.
"""

from urllib.parse import quote

import httpx2

from llm_gateway.agents.world import ToolOutcome
from llm_gateway.connectors.base import (
    ConnectorStatus,
    capped,
    dump,
    error,
    function,
    invalid,
)

USER_AGENT = "local-llm-gateway/1.0"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
FX = "https://api.frankfurter.dev/v1/latest"
WIKIPEDIA = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# WMO weather codes used by Open-Meteo.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}  # fmt: skip


class WebApis:
    name = "web"
    label = "Web APIs"

    def __init__(self, enabled: bool, client: httpx2.AsyncClient | None = None) -> None:
        self.enabled = enabled
        self._client = client

    def status(self) -> ConnectorStatus:
        if not self.enabled:
            return ConnectorStatus(self.name, self.label, False, "Switched off (GATEWAY_WEB_APIS).")
        return ConnectorStatus(
            self.name, self.label, True, "Open-Meteo weather, Frankfurter rates, Wikipedia"
        )

    def tools(self) -> list[dict]:
        return [
            function(
                "get_weather",
                "Current weather for a city: temperature, conditions and wind.",
                {"city": {"type": "string"}},
                ["city"],
            ),
            function(
                "convert_currency",
                "Convert an amount between currencies at today's reference rate (ECB).",
                {
                    "amount": {"type": "number"},
                    "from": {"type": "string", "description": "ISO code, e.g. EUR"},
                    "to": {"type": "string", "description": "ISO code, e.g. USD"},
                },
                ["amount", "from", "to"],
            ),
            function(
                "wikipedia_summary",
                "The opening summary of an English Wikipedia article.",
                {"topic": {"type": "string", "description": "Article title, e.g. Ada Lovelace"}},
                ["topic"],
            ),
        ]

    async def _get(self, url: str, params: dict | None = None) -> tuple[int, dict]:
        client = self._client or httpx2.AsyncClient()
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=15.0,
                follow_redirects=True,
            )
        finally:
            if self._client is None:
                await client.aclose()
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {}

    async def call(self, name: str, args: dict) -> ToolOutcome:
        if not self.enabled:
            return error("Web APIs are switched off.")
        try:
            if name == "get_weather":
                return await self._weather(str(args.get("city") or "").strip())
            if name == "convert_currency":
                return await self._convert(args)
            if name == "wikipedia_summary":
                return await self._wikipedia(str(args.get("topic") or "").strip())
        except httpx2.HTTPError as exc:
            return error(f"The service could not be reached: {type(exc).__name__}")
        return invalid(f"Unknown tool: {name}")

    async def _weather(self, city: str) -> ToolOutcome:
        if not city:
            return invalid("Missing required argument: city")
        status, places = await self._get(GEOCODE, {"name": city, "count": 1})
        found = (places.get("results") or [None])[0] if status == 200 else None
        if not found:
            return error(f"Unknown city: {city}")
        status, data = await self._get(
            FORECAST,
            {
                "latitude": found["latitude"],
                "longitude": found["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )
        now = data.get("current") if status == 200 else None
        if not now:
            return error("The weather service did not answer.")
        return ToolOutcome(
            dump(
                {
                    "city": found["name"],
                    "country": found.get("country"),
                    "time": now.get("time"),
                    "temperature_c": now.get("temperature_2m"),
                    "conditions": WEATHER_CODES.get(now.get("weather_code"), "unknown"),
                    "wind_kmh": now.get("wind_speed_10m"),
                }
            )
        )

    async def _convert(self, args: dict) -> ToolOutcome:
        missing = [k for k in ("amount", "from", "to") if args.get(k) in (None, "")]
        if missing:
            return invalid(f"Missing required argument(s): {', '.join(missing)}")
        try:
            amount = float(args["amount"])
        except (TypeError, ValueError):
            return invalid("amount must be a number")
        source, target = str(args["from"]).upper().strip(), str(args["to"]).upper().strip()
        status, data = await self._get(FX, {"amount": amount, "from": source, "to": target})
        if status != 200 or target not in (data.get("rates") or {}):
            return error(f"No rate for {source} to {target}.")
        return ToolOutcome(
            dump(
                {
                    "amount": amount,
                    "from": source,
                    "to": target,
                    "result": data["rates"][target],
                    "rate_date": data.get("date"),
                }
            )
        )

    async def _wikipedia(self, topic: str) -> ToolOutcome:
        if not topic:
            return invalid("Missing required argument: topic")
        status, data = await self._get(WIKIPEDIA + quote(topic.replace(" ", "_"), safe=""))
        if status != 200 or not data.get("extract"):
            return error(f"No Wikipedia article found for: {topic}")
        return ToolOutcome(capped(dump({"title": data.get("title"), "summary": data["extract"]})))
