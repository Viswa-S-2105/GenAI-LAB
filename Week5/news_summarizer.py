import os
import feedparser
import google.generativeai as genai

# ✅ Configure the Gemini client
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Load the Gemini 1.5 Pro model
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")


# ✅ Fetch articles from an RSS feed
def get_rss_articles(feed_url, max_articles=3):
    feed = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "summary": entry.summary if 'summary' in entry else "",
        })
    return articles


# ✅ Summarization prompt templates
def get_prompt(content, mode):
    if mode == 'brief':
        return f"Summarize the following news article in 2-3 concise sentences:\n\n{content}"
    elif mode == 'detailed':
        return f"Write a detailed summary of the following news article with all key points:\n\n{content}"
    elif mode == 'bullet-points':
        return f"Summarize the following article into bullet points:\n\n{content}"
    else:
        raise ValueError("Unsupported mode. Choose from 'brief', 'detailed', or 'bullet-points'.")


# ✅ Generate summary using Gemini
def summarize_article(article, mode='brief'):
    prompt = get_prompt(article["summary"], mode)
    response = model.generate_content(prompt)
    return response.text.strip()


# ✅ Run summarizer
def summarize_rss_feed(feed_url, mode='brief', max_articles=3):
    articles = get_rss_articles(feed_url, max_articles)
    summaries = []

    for article in articles:
        summary = summarize_article(article, mode)
        summaries.append({
            "title": article["title"],
            "link": article["link"],
            "summary": summary
        })

    return summaries


# ✅ Example usage
if __name__ == "__main__":
    rss_url = "https://rss.cnn.com/rss/edition.rss"  # or any other RSS feed
    mode = "bullet-points"  # brief, detailed, bullet-points
    results = summarize_rss_feed(rss_url, mode=mode)

    for i, res in enumerate(results, 1):
        print(f"\n📰 Article {i}: {res['title']}")
        print(f"🔗 {res['link']}")
        print(f"📝 Summary:\n{res['summary']}")
