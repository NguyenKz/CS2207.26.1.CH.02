"""Browser acceptance checks for the Dataset subtab inside Demo.

Run with the FastAPI backend on port 6788 and Vite on port 5113, for example:
source .venv/bin/activate && python /path/to/with_server.py \
--server "uvicorn gk.web.backend.server:app --port 6788" --port 6788 \
  --server "cd gk/web/frontend && npm run dev -- --host 127.0.0.1 --port 5113" --port 5113 \
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
    page.get_by_role("button", name="Demo", exact=True).click()
    page.get_by_role("tab", name="DATASET", exact=True).click()


def assert_no_horizontal_overflow(page: Page) -> None:
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_review_tab(page: Page) -> None:
    print("UI: review tab")
    page.set_default_timeout(8_000)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    assert page.get_by_role("button", name="Inspect", exact=True).count() == 0
    assert page.get_by_role("button", name="Dataset", exact=True).count() == 0
    page.get_by_role("button", name="Demo", exact=True).click()
    assert page.get_by_role("tab", name="PREDICTION BOARD", exact=True).count() == 1
    assert page.get_by_role("tab", name="DATASET", exact=True).count() == 1
    page.get_by_role("button", name="Review", exact=True).click()
    assert page.locator(".activation-card").count() == 5
    assert page.get_by_text("Total parameters", exact=True).count() == 1
    assert "Softplus" not in page.locator("body").inner_text()
    assert "Delay per epoch" not in page.locator("body").inner_text()
    assert "Final %" not in page.locator("body").inner_text()
    assert "Holdout" not in page.locator("body").inner_text()
    parameters = page.locator(".parameter-counter strong")
    assert parameters.inner_text() == "1,386"
    page.locator('select[aria-label="Input feature count"]').select_option("8")
    assert parameters.inner_text() == "618"
    page.locator('select[aria-label="Hidden neuron count"]').select_option("64")
    assert parameters.inner_text() == "1,226"
    for width, height in ((360, 800), (768, 1024), (1280, 900)):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(100)
        assert_no_horizontal_overflow(page)


def test_review_training(page: Page) -> None:
    print("UI: review training")
    page.set_default_timeout(30_000)
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Review", exact=True).click()
    page.get_by_label("Total", exact=True).fill("100")
    page.get_by_label("Epochs", exact=True).fill("1")
    page.get_by_role("button", name="Start training", exact=True).click()
    page.get_by_text("Run complete", exact=True).wait_for()
    assert page.locator(".summary-row").count() == 5

    page.get_by_role("button", name="Reset", exact=True).click()
    page.get_by_label("Total", exact=True).fill("100")
    page.get_by_label("Epochs", exact=True).fill("5000")
    page.get_by_role("button", name="Start training", exact=True).click()
    page.get_by_text("Training realtime", exact=True).wait_for()
    page.get_by_role("button", name="Cancel", exact=True).click()
    page.get_by_text("Run cancelled", exact=True).wait_for()


def test_successful_browse(page: Page) -> None:
    print("UI: success browse")
    page.set_default_timeout(8_000)
    requests: list[str] = []
    console_errors: list[str] = []
    page.on("request", lambda request: requests.append(request.url) if "/dataset/" in request.url else None)
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)

    open_dataset(page)
    page.wait_for_selector(".dataset-sample-grid")
    assert page.get_by_text("Quick look at all digits", exact=True).count() == 0
    assert page.get_by_text("SELECTED SAMPLE", exact=True).count() == 0
    assert page.locator(".dataset-sample-grid .dataset-digit-tile").count() == 60
    assert page.locator(".dataset-sample-grid img[loading='lazy']").count() == 60
    assert page.locator(".dataset-sample-grid .dataset-digit-tile").count() < 100
    assert any("/dataset/predict/samples" in url and "limit=60" in url for url in requests)
    page.screenshot(path=SCREENSHOT_DIR / "predict-overview.png", full_page=True)

    class_button = page.locator(".dataset-class-button").nth(8)
    class_button.focus()
    page.keyboard.press("Enter")
    page.wait_for_function("document.querySelectorAll('.dataset-sample-grid .dataset-digit-tile').length === 60")
    assert any("label=7" in url and "limit=60" in url for url in requests)
    page.screenshot(path=SCREENSHOT_DIR / "predict-class-grid.png", full_page=True)

    page.locator(".dataset-sample-grid .dataset-digit-tile").first.click()
    assert page.locator(".dataset-sample-grid .dataset-digit-tile.is-selected").count() == 1
    page.screenshot(path=SCREENSHOT_DIR / "predict-detail.png", full_page=True)
    browser_panel = page.locator(".dataset-browser-panel")
    browser_panel.hover()
    page.mouse.wheel(0, 5_000)
    page.wait_for_function("document.querySelectorAll('.dataset-sample-grid .dataset-digit-tile').length >= 120")

    for width, height in ((360, 800), (390, 844), (768, 1024), (1280, 900), (1440, 900)):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(150)
        assert_no_horizontal_overflow(page)
        page.screenshot(path=SCREENSHOT_DIR / f"dataset-{width}.png", full_page=True)

    assert not console_errors, console_errors


def test_error_and_empty_states(page: Page) -> None:
    print("UI: error and empty states")
    page.set_default_timeout(8_000)
    page.route("**/dataset/predict/meta", lambda route: route.fulfill(status=503, content_type="application/json", body='{"detail":"simulated dataset outage"}'))
    open_dataset(page)
    page.wait_for_selector(".dataset-state-error")
    assert page.get_by_text("simulated dataset outage", exact=True).count() == 1
    page.unroute("**/dataset/predict/meta")
    page.get_by_role("button", name="Retry").click()
    page.wait_for_selector(".dataset-sample-grid")

    page.route("**/dataset/predict/samples**", lambda route: route.fulfill(status=200, content_type="application/json", body='{"items":[],"offset":0,"limit":40,"total":0,"has_more":false}'))
    page.reload()
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Demo", exact=True).click()
    page.get_by_role("tab", name="DATASET", exact=True).click()
    page.get_by_text("No samples", exact=True).wait_for()
    assert page.get_by_text("No samples", exact=True).count() == 1


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 900})
        test_review_tab(context.new_page())
        test_review_training(context.new_page())
        test_successful_browse(context.new_page())
        test_error_and_empty_states(context.new_page())
        browser.close()
    print(f"PASS: Dataset UI acceptance checks and screenshots in {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()
