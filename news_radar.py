import os

from core import run_ai_news_radar, send_new_ai_radar_alerts


def main() -> None:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    payload = run_ai_news_radar(api_key, include_gdelt=True)

    alerts = 0
    if os.getenv("ENABLE_RADAR_ALERTS", "true").lower() == "true":
        alerts = send_new_ai_radar_alerts(
            payload,
            app_url=os.getenv("APP_URL", ""),
            minimum_impact=float(os.getenv("RADAR_ALERT_THRESHOLD", "85")),
        )

    print(
        f'AI News Radar: {payload.get("event_count", 0)} events, '
        f'{payload.get("emerging_signal_count", 0)} emerging signals, '
        f'{alerts} alerts sent.'
    )


if __name__ == "__main__":
    main()
