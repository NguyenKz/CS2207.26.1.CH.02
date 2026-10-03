from playwright.sync_api import sync_playwright


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:5173", wait_until="networkidle")
    page.get_by_role("button", name="Demo").click()
    random_button = page.get_by_role("button", name="Random sample")
    random_button.click()
    page.get_by_role("button", name="Predicting…").wait_for(state="attached", timeout=5_000)
    page.get_by_text("Run Predict to see output probabilities.").first.wait_for(state="detached", timeout=15_000)
    assert page.locator(".predict-model-result").count() == 4
    print("PASS: Random sample automatically predicts")
    browser.close()
