"""Browser acceptance checks for the Dataset tab.

Run with the FastAPI backend on port 6788 and Vite on port 5113, for example:
source .venv/bin/activate && python /path/to/with_server.py \
  --server "uvicorn gk.web.backend.server:app --port 6788" --port 6788 \
  --server "npm run dev -- --host 127.0.0.1 --port 5113" --port 5113 \
  -- python gk/web/frontend/tests/dataset_ui.py
"""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = "http://localhost:5113"
SCREENSHOT_DIR = Path(os.environ.get("DATASET_SCREENSHOT_DIR", "/tmp/ann-dataset-screens"))


def open_dataset(page: Page) -> None:
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Dataset", exact=True).click()


def assert_no_horizontal_overflow(page: Page) -> None:
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_successful_browse(page: Page) -> None:
    print("UI: success browse")
    page.set_default_timeout(8_000)
    requests: list[str] = []
    console_errors: list[str] = []
    page.on("request", lambda request: requests.append(request.url) if "/dataset/" in request.url else None)
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)

    open_dataset(page)
    page.wait_for_selector(".dataset-representative-grid")
    assert page.locator(".dataset-representative-grid .dataset-digit-tile").count() == 10
    assert any("/dataset/predict/samples" in url and "limit=40" in url for url in requests)
    page.screenshot(path=SCREENSHOT_DIR / "predict-overview.png", full_page=True)

    class_button = page.locator(".dataset-class-button").nth(8)
    class_button.focus()
    page.keyboard.press("Enter")
    page.wait_for_function("document.querySelectorAll('.dataset-sample-grid .dataset-digit-tile').length === 40")
    assert any("label=7" in url and "limit=40" in url for url in requests)
    page.screenshot(path=SCREENSHOT_DIR / "predict-class-grid.png", full_page=True)

    page.locator(".dataset-sample-grid .dataset-digit-tile").first.click()
    assert page.get_by_text("Test sample #", exact=False).count() == 1
    page.screenshot(path=SCREENSHOT_DIR / "predict-detail.png", full_page=True)
    load_more = page.get_by_role("button", name="Load more samples")
    if load_more.count():
        load_more.click()
        page.wait_for_function("document.querySelectorAll('.dataset-sample-grid .dataset-digit-tile').length >= 80")

    page.get_by_role("tab", name="Train dataset").click()
    page.wait_for_selector(".dataset-scatter")
    page.wait_for_selector(".dataset-sample-table tbody tr")
    assert page.locator(".dataset-class-legend button").count() == 10
    assert page.locator(".dataset-feature-selectors select").first.locator("option:disabled").count() == 1
    page.locator(".dataset-feature-selectors select").first.select_option("2")
    page.wait_for_timeout(500)
    page.locator(".dataset-class-legend button").first.press("Space")
    row = page.locator(".dataset-sample-table tbody tr").first
    row.focus()
    row.press("Enter")
    assert page.get_by_text("SELECTED VECTOR", exact=True).count() == 1
    page.screenshot(path=SCREENSHOT_DIR / "train-scatter-table.png", full_page=True)

    for width, height in ((360, 800), (390, 844), (768, 1024), (1280, 900), (1440, 900)):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(150)
        assert_no_horizontal_overflow(page)
        page.screenshot(path=SCREENSHOT_DIR / f"train-{width}.png", full_page=True)

    assert not console_errors, console_errors


def test_error_and_empty_states(page: Page) -> None:
    print("UI: error and empty states")
    page.set_default_timeout(8_000)
    page.route("**/dataset/predict/meta", lambda route: route.fulfill(status=503, content_type="application/json", body='{"detail":"simulated dataset outage"}'))
    open_dataset(page)
    page.wait_for_selector(".dataset-state-error")
    assert page.get_by_text("simulated dataset outage", exact=True).count() == 1
    page.unroute("**/dataset/predict/meta")
    page.get_by_role("button", name="Thử lại").click()
    page.wait_for_selector(".dataset-representative-grid")

    page.route("**/dataset/predict/samples**", lambda route: route.fulfill(status=200, content_type="application/json", body='{"items":[],"offset":0,"limit":40,"total":0,"has_more":false}'))
    page.reload()
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Dataset", exact=True).click()
    page.get_by_text("Không có mẫu", exact=True).wait_for()
    assert page.get_by_text("Không có mẫu", exact=True).count() == 1


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 900})
        test_successful_browse(context.new_page())
        test_error_and_empty_states(context.new_page())
        browser.close()
    print(f"PASS: Dataset UI acceptance checks and screenshots in {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
