"""
Constants for experiments.

"""

HF_DATASET_CACHE_DIR = "hf_dataset_cache"

# Map domain names to dataset subset names in the corresponding Hugging Face dataset.
# Dataset: amazon_us_reviews
AMAZON_REVIEWS_CATEGORY_SUBSETS = {
    "apparel": ["Apparel_v1_00"], "automotive": ["Automotive_v1_00"],
    "baby": ["Baby_v1_00"], "beauty": ["Beauty_v1_00"],
    "books": ["Books_v1_00", "Books_v1_01", "Books_v1_02"], "camera": ["Camera_v1_00"],
    "digital_ebook_purchase": ["Digital_Ebook_Purchase_v1_00", "Digital_Ebook_Purchase_v1_01"],
    "digital_music_purchase": ["Digital_Music_Purchase_v1_00"],
    "digital_software": ["Digital_Software_v1_00"], "digital_video_download": ["Digital_Video_Download_v1_00"],
    "digital_video_games": ["Digital_Video_Games_v1_00"], "electronics": ["Electronics_v1_00"],
    "furniture": ["Furniture_v1_00"], "gift_card": ["Gift_Card_v1_00"], "grocery": ["Grocery_v1_00"],
    "health_personal_care": ["Health_Personal_Care_v1_00"], "home_entertainment": ["Home_Entertainment_v1_00"],
    "home_improvement": ["Home_Improvement_v1_00"], "home": ["Home_v1_00"],
    "jewelry": ["Jewelry_v1_00"], "kitchen": ["Kitchen_v1_00"],
    "lawn_and_garden": ["Lawn_and_Garden_v1_00"], "luggage": ["Luggage_v1_00"],
    "major_appliances": ["Major_Appliances_v1_00"], "mobile_apps": ["Mobile_Apps_v1_00"],
    "mobile_electronics": ["Mobile_Electronics_v1_00"], "music": ["Music_v1_00"],
    "musical_instruments": ["Musical_Instruments_v1_00"], "office_products": ["Office_Products_v1_00"],
    "outdoors": ["Outdoors_v1_00"], "pc": ["PC_v1_00"],
    "personal_care_appliances": ["Personal_Care_Appliances_v1_00"], "pet_products": ["Pet_Products_v1_00"],
    "shoes": ["Shoes_v1_00"], "software": ["Software_v1_00"], "sports": ["Sports_v1_00"],
    "tools": ["Tools_v1_00"], "toys": ["Toys_v1_00"], "video_dvd": ["Video_DVD_v1_00"],
    "video_games": ["Video_Games_v1_00"], "video": ["Video_v1_00"],
    "watches": ["Watches_v1_00"], "wireless": ["Wireless_v1_00"]
}

AMAZON_REVIEWS_YEARS = list(range(2001, 2016))

MNLI_TRAIN_GENRES = ["fiction", "government", "slate", "telephone", "travel"]
MNLI_EVAL_GENRES = ["facetoface", "fiction", "government", "letters", "nineeleven", "oup", "slate", "telephone", "travel", "verbatim"]

# From the Universal POS tags, with added spaCy tags and SEP for sentence boundaries.
SPACY_POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
                  "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] + ["SPACE", "SEP"]
CONTENT_POS_TAGS = ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]
