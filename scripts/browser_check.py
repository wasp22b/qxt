"""Optional UI smoke test: pip install playwright; playwright install chromium."""
from playwright.sync_api import sync_playwright
from pathlib import Path
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    page.wait_for_selector('#kpis .metric')
    assert 'Protected' in page.locator('#kpis').inner_text()
    page.screenshot(path='results/dashboard.png',full_page=True)
    page.select_option('#stress','grid_outage');page.click('#run')
    page.wait_for_function("document.querySelector('#run').disabled === false")
    assert 'shortages' in page.locator('#safety').inner_text()
    page.locator('[data-page="quantum"]').click();page.click('#qrun')
    page.wait_for_selector('#qresults table',timeout=90000)
    assert page.locator('#qresults tbody tr').count()==3
    page.screenshot(path='results/quantum-dashboard.png',full_page=True)
    page.select_option('#qscope','slot');page.click('#qrun')
    page.wait_for_selector('#recourse-export',timeout=90000)
    assert 'Full-city candidate validation' in page.locator('#qresults').inner_text()
    page.screenshot(path='results/quantum-city-recourse.png',full_page=True)
    page.locator('[data-page="configure"]').click()
    page.locator('[data-source="2"][data-key="cost"]').fill('8')
    page.click('#apply');page.wait_for_function("document.querySelector('#apply').disabled === false")
    assert 'Updated constraints applied' in page.locator('#message').inner_text()
    page.click('[data-page="overview"]');page.click('#compare')
    page.wait_for_function("document.querySelector('#compare').disabled === false",timeout=90000)
    assert page.locator('#comparisons tbody tr').count()>=5
    with page.expect_download() as dl:
        page.click('#export')
    assert dl.value.suggested_filename=='quantum-city-plan.json'
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth')<=395
    page.screenshot(path='results/mobile-dashboard.png',full_page=True)
    assert not errors,errors
    print('Browser passed: initial dispatch, shortage scenario, Aer quantum job, edited costs, scenario comparison, mobile layout; no JS errors.')
    browser.close()
