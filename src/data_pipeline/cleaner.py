import re
import unicodedata


class Cleaner:

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize Vietnamese text"""
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text)

        text = "".join(
            char for char in text
            if not unicodedata.category(char).startswith("C") or char in ["\n", "\t"]
        )

        text = re.sub(r'[^\w\s.,!?;:()\-\*]', '', text)

        return text.strip()

    @staticmethod
    def parse_stars(stars: str) -> float:
        """Convert '5/5' -> 5"""
        try:
            return float(stars.split("/")[0])
        except:
            return 0

    @staticmethod
    def dict_to_text(item: dict) -> str:
        """Convert dict to formatted text"""

        name = Cleaner.clean_text(item.get("location_name", "Unknown Place"))
        category = Cleaner.clean_text(item.get("category", "Unknown Category"))
        address = Cleaner.clean_text(item.get("address", "No Address Provided"))
        rating = item.get("overall_rating", "N/A")
        rating_count = item.get("rating_count", "0")
        description = Cleaner.clean_text(item.get("description", "No Description"))
        url = item.get("url", "")

        reviews = item.get("reviews", [])

        top_reviews = sorted(
            reviews,
            key=lambda r: Cleaner.parse_stars(r.get("stars", "0/5")),
            reverse=True
        )[:3]

        reviews_text = ""

        for review in top_reviews:
            stars = review.get("stars", "N/A")
            comment = Cleaner.clean_text(review.get("comment", "No comment"))

            reviews_text += f'* "{comment}" ({stars})\n'

        text = f"""
Location: {name}
Category: {category}
Address: {address}
Rating: {rating} ({rating_count} reviews)

Description:
{description}

Top Reviews:
{reviews_text if reviews_text else "No reviews available."}

URL: {url}
"""

        return text.strip()

    @classmethod
    def json_to_markdown(cls, item: dict) -> str:
        """Convert JSON dict -> markdown"""

        text = cls.dict_to_text(item)

        markdown = f"""# Location Information
{text}
"""
        return markdown.strip()