import asyncio
import os
import re
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from openai import OpenAI


# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)


# -----------------------------
# Step 1: Scrape Website
# -----------------------------
async def scrape_website(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            user_agent="Mozilla/5.0"
        )

        page = await context.new_page()

        await page.goto(url, wait_until="networkidle")

        # Scroll to load dynamic content
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)

        html = await page.content()
        await browser.close()

        return html


# -----------------------------
# Step 2: Extract meaningful text
# -----------------------------
def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    elements = soup.find_all(["h1", "h2", "h3", "p", "li"])

    text_list = []
    for el in elements:
        txt = el.get_text(strip=True)

        if len(txt) > 30:
            text_list.append(txt)

    content = " ".join(text_list)

    # reduce size to avoid token limit
    return content[:5000]


# -----------------------------
# Step 3: Groq summarization
# -----------------------------
def summarize_with_groq(text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that summarizes website content clearly "
                    "in structured bullet points."
                )
            },
            {
                "role": "user",
                "content": f"""
Summarize the following website.

Focus on:
- About the person
- Skills
- Projects
- Experience

Content:
{text}
"""
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# -----------------------------
# Step 4: Main pipeline
# -----------------------------
async def main():
    url = "https://portfolio-sm-iota.vercel.app/"

    print("🔍 Scraping website...")
    html = await scrape_website(url)

    print("🧹 Extracting content...")
    clean_text = extract_visible_text(html)

    print("\n🤖 Generating summary with Groq...\n")
    summary = summarize_with_groq(clean_text)

    print("📌 SUMMARY:\n")
    print(summary)


# Run
if __name__ == "__main__":
    asyncio.run(main())
