"""Scripted walkthrough of the dashboard (Playwright + Chrome); captures timestamped screenshots.
import asyncio
import os
import sys
import time

from playwright.async_api import async_playwright

OUT = sys.argv[1]
URL = "http://localhost:8777/"
W, H = 1440, 1160

CURSOR_JS = """
(() => {
  const c = document.createElement('div');
  c.id = '__cur';
  c.style.cssText = 'position:fixed;left:0;top:0;width:18px;height:18px;z-index:99999;pointer-events:none;' +
    'transform:translate(-2px,-2px);transition:transform .05s';
  c.innerHTML = '<svg width="18" height="18" viewBox="0 0 18 18"><path d="M2 1l13 7-6 1.5L6 16z" fill="white" stroke="black" stroke-width="1.2"/></svg>';
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', () => { c.firstChild.style.transform = 'scale(.8)'; }, true);
  document.addEventListener('mouseup', () => { c.firstChild.style.transform = ''; }, true);
})();
"""


async def main():
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True,
                                    args=["--autoplay-policy=no-user-gesture-required"])
        page = await b.new_page(viewport={"width": W, "height": H})
        await page.goto(URL)
        await page.wait_for_timeout(1500)
        await page.evaluate(CURSOR_JS)
        await page.mouse.move(700, 450)

        frames = []
        rec = {"on": True}

        async def capture():
            i = 0
            while rec["on"]:
                t = time.time()
                path = os.path.join(OUT, f"f{i:05d}.jpg")
                await page.screenshot(path=path, type="jpeg", quality=90)
                frames.append((path, t))
                i += 1

        async def glide(x, y, steps=12):
            await page.mouse.move(x, y, steps=steps)
            await page.wait_for_timeout(150)

        async def click(sel, pause=600):
            box = await page.locator(sel).first.bounding_box()
            await glide(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            await page.mouse.down()
            await page.mouse.up()
            await page.wait_for_timeout(pause)

        async def wait_ready():
            await page.wait_for_function(
                "[...document.querySelectorAll('video')].filter(v=>v.offsetParent).every(v=>v.readyState>=3)",
                timeout=15000)

        cap = asyncio.create_task(capture())
        await page.select_option("#speed", "3")
        await page.wait_for_timeout(800)

        await page.evaluate("document.querySelector('video').currentTime = 16")
        # 1) hot seed 1: play
        await wait_ready()
        await click("#play", 4500)
        # 2) switch to cold seed 1
        await click('[data-run="cold_s1"]', 400)
        await wait_ready()
        await page.evaluate("document.querySelector('video').currentTime = 20")
        await click("#play", 3500)
        # 3) immunoediting seed 3, lineage view
        await click('[data-run="edited_s3"]', 400)
        await wait_ready()
        await click('#viewopt button[data-v="lineage"]', 300)
        await wait_ready()
        await page.evaluate("document.querySelector('video').currentTime = 35")
        await click("#play", 3500)
        await click("#play", 300)  # pause
        # 4) hover the tumor chart, click to seek
        box = await page.locator("svg.plot").first.bounding_box()
        y = box["y"] + box["height"] * 0.5
        await glide(box["x"] + 20, y)
        for k in range(1, 21):
            await page.mouse.move(box["x"] + 20 + k * (box["width"] - 40) / 20, y)
            await page.wait_for_timeout(70)
        await page.mouse.down()
        await page.mouse.up()
        await page.wait_for_timeout(1200)
        # 5) compare: treatment timing
        await click("#m-compare", 500)
        await click('#cmpbar .presets button:nth-of-type(2)', 400)
        await wait_ready()
        await page.evaluate("document.querySelectorAll('video').forEach(v => v.currentTime = 20)")
        await page.wait_for_timeout(400)
        await click("#play", 5000)
        # 6) all treatment scenarios
        await click('#cmpbar .presets button:last-child', 400)
        await wait_ready()
        await page.evaluate("document.querySelectorAll('video').forEach(v => v.currentTime = 26)")
        await page.wait_for_timeout(400)
        await click("#play", 5000)

        rec["on"] = False
        await cap
        await b.close()
        with open(os.path.join(OUT, "list.txt"), "w") as f:
            for (pth, t), (_, t2) in zip(frames, frames[1:] + [(None, frames[-1][1] + 0.1)]):
                f.write(f"file '{os.path.basename(pth)}'\nduration {t2 - t:.3f}\n")
            f.write(f"file '{os.path.basename(frames[-1][0])}'\n")
        span = frames[-1][1] - frames[0][1]
        print(f"{len(frames)} frames over {span:.1f}s ({len(frames) / span:.1f} fps)")


asyncio.run(main())
