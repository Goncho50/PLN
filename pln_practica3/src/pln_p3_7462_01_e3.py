import json
import google.generativeai as genai

genai.configure(api_key="AIzaSyDUsxQN9uInH9wm9uyjWrGdui6aBtlmrdI")
model = genai.GenerativeModel("gemini-2.5-flash")

game_id = 16398

# Filter reviews for the selected game only
with open("corpus_features_balanced.json", "r", encoding="utf-8") as f:
    reviews_json = json.load(f)

game_reviews = [r for r in reviews_json if r.get("game_id") == game_id]
reviews_str = json.dumps(game_reviews, ensure_ascii=False)

prompt = f"""
You are given a JSON object containing multiple reviews for a game (game_id: {game_id}).
Analyze all reviews and produce a concise summary.

Tasks:
1) List up to 5 common positive points (bullet list)
2) List up to 5 common negative points (bullet list)
3) Contrary opinions between positive points and negative points (if any)
4) Overall sentiment: Positive, Neutral, or Negative

Output in plain text:
Game: {game_id}
Summary:
- Positive: ...
- Negative: ...
- Contrary Opinions: ...
Overall Sentiment: ...

JSON reviews:
{reviews_str}
"""

response = model.generate_content(prompt)
print(response.text)
