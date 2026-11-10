"""
Sahaya Emergency Assistant (Gradio on Render)
High-Speed Voice-First Emergency Triage & First Aid for India.
"""

import base64
import html
import json
import math
import os
import re
import tempfile
import gradio as gr
import numpy as np
import requests
import soundfile as sf
from scipy.signal import butter, sosfilt

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

TARGET_SR = 16000
MAX_SECONDS = 30

HTTP_SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
HTTP_SESSION.mount("https://", adapter)
HTTP_SESSION.mount("http://", adapter)

# ---------------------------------------------------------------------------
# Supported Languages
# ---------------------------------------------------------------------------
LANGUAGES = {
    "en_IN": "English (India)",
    "kn_IN": "ಕನ್ನಡ (Kannada)",
    "hi_IN": "हिन्दी (Hindi)",
    "ta_IN": "தமிழ் (Tamil)",
    "te_IN": "తెలుగు (Telugu)",
    "ml_IN": "മലയാളം (Malayalam)",
    "mr_IN": "मराठी (Marathi)",
    "gu_IN": "ગુજરાતી (Gujarati)",
    "bn_IN": "বাংলা (Bengali)",
    "pa_IN": "ਪੰਜਾਬੀ (Punjabi)",
    "or_IN": "ଓଡ଼ିଆ (Odia)",
    "mni_IN": "ꯃꯤꯇꯩꯂꯣꯟ (Manipuri)",
}

SCRIPT_NAMES = {
    "en_IN": "English",
    "kn_IN": "Kannada script (ಕನ್ನಡ)",
    "hi_IN": "Hindi, Devanagari script (हिन्दी)",
    "ta_IN": "Tamil script (தமிழ்)",
    "te_IN": "Telugu script (తెలుగు)",
    "ml_IN": "Malayalam script (മലയാളം)",
    "mr_IN": "Marathi, Devanagari script (मराठी)",
    "gu_IN": "Gujarati script (ગુજરાતી)",
    "bn_IN": "Bengali script (বাংলা)",
    "pa_IN": "Gurmukhi script (ਪੰਜਾਬੀ)",
    "or_IN": "Odia script (ଓଡ଼ିଆ)",
    "mni_IN": "Meitei Mayek script (ꯃꯤꯇꯩꯂꯣꯟ)",
}

TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

WHISPER_NAME_TO_CODE = {
    "english": "en_IN", "hindi": "hi_IN", "urdu": "hi_IN", "nepali": "hi_IN",
    "kannada": "kn_IN", "tamil": "ta_IN", "telugu": "te_IN", "malayalam": "ml_IN",
    "marathi": "mr_IN", "gujarati": "gu_IN", "bengali": "bn_IN", "assamese": "bn_IN",
    "punjabi": "pa_IN", "odia": "or_IN", "oriya": "or_IN", "manipuri": "mni_IN",
    "en": "en_IN", "hi": "hi_IN", "ur": "hi_IN", "kn": "kn_IN", "ta": "ta_IN",
    "te": "te_IN", "ml": "ml_IN", "mr": "mr_IN", "gu": "gu_IN", "bn": "bn_IN",
    "as": "bn_IN", "pa": "pa_IN", "or": "or_IN", "mni": "mni_IN",
}

SCRIPT_RANGES = [
    (0x0C80, 0x0CFF, "kn_IN"),
    (0x0900, 0x097F, "hi_IN"),
    (0x0B80, 0x0BFF, "ta_IN"),
    (0x0C00, 0x0C7F, "te_IN"),
    (0x0D00, 0x0D7F, "ml_IN"),
    (0x0980, 0x09FF, "bn_IN"),
    (0x0A80, 0x0AFF, "gu_IN"),
    (0x0A00, 0x0A7F, "pa_IN"),
    (0x0B00, 0x0B7F, "or_IN"),
    (0xABC0, 0xABFF, "mni_IN"),
]

MR_MARKERS = ["आहे", "आहेत", "मला", "माझ", "तुम्ही", "करा", "झाले", "झाला", "नाही", "छातीत", "दुखत"]
HI_MARKERS = ["है", "हैं", "मुझे", "मेरे", "मेरी", "में", "रहा", "रही", "नहीं", "करो", "कृपया", "मदद"]

ROMAN_LANG_KEYWORDS = {
    "kn_IN": {"nanage", "nanna", "namma", "tande", "taayi", "ede", "edenovu", "novu", "usiru", "aagide", "beku", "bega", "maadi", "sahaya", "rakta", "aspatre", "kattada", "bidda", "haavu"},
    "hi_IN": {"madad", "chahiye", "bachao", "bachaao", "mujhe", "humein", "mere", "mera", "meri", "dard", "saans", "takleef", "saanp", "khoon", "aspatal", "gira", "hai", "hain", "jaldi", "behosh"},
    "ta_IN": {"udavi", "enakku", "nenju", "vali", "swasam", "moochu", "pambu", "ratham", "maruthuvamanai", "vizhunthuttar", "irukku", "seekiram"},
    "te_IN": {"sahayam", "naaku", "gunde", "noppi", "swasa", "kastam", "paamu", "aaspatri", "kinda", "undi", "ledu", "tvaraga"},
    "ml_IN": {"sahayam", "enikku", "vedana", "shwasam", "muttal", "paampu", "aashupathri", "veenu", "undu", "vegam"},
    "mr_IN": {"madat", "mala", "majhya", "dukhate", "dukhat", "trass", "davakhana", "padla", "aahe", "aahet", "lavkar"},
    "gu_IN": {"madad", "mane", "maru", "mara", "lohi", "padi", "che", "nathi", "tarat"},
    "bn_IN": {"sahajjo", "amar", "amake", "buker", "byatha", "kosto", "rokto", "pore", "ache", "taratari"},
    "pa_IN": {"madad", "sanu", "mainu", "peerh", "saah", "aukh", "diga"},
    "or_IN": {"sahajya", "mora", "bitha", "shwasa", "kasta", "padigala"},
    "mni_IN": {"mateng", "eigi", "thamoigi", "chikpa", "masha", "lin"},
}

EN_COMMON = {"the", "is", "help", "emergency", "accident", "fall", "collapse", "building", "hospital", "doctor", "ambulance", "police", "chest", "pain", "fire", "bleeding", "blood", "head", "injury", "broken", "breath", "breathing", "heart", "attack", "someone", "please", "urgent", "life", "died", "unconscious"}

BASE_T = {
    "brand": "Sahaya",
    "ready_status": "Ready",
    "hero_title": "AI Voice Assistant",
    "subtitle": "Talk. Translate. Stay Safe.",
    "sos_btn": "🚨 EMERGENCY SOS • LOUD ALARM + DIAL 112",
    "mic_header": "Tap to Start Recording",
    "mic_recording": "Recording... (Speak now)",
    "mic_sub": "Speak naturally in any Indian language",
    "shield_text": "Real-time emergency triage & first aid",
    "auto_detect_badge": "Auto-Detect Active (12 Indian Languages)",
    "placeholder": "Or type emergency details (e.g., chest pain, accident)...",
    "send_btn": "Translate & Help",
    "clear_btn": "🗑️ Clear",
    "live_trans": "Live Emergency Triage",
    "notified_sub": "Emergency Services Notified",
    "first_aid_label": "Immediate First-Aid Steps:",
    "call_btn": "Call {number} Immediately",
    "directory_header": "Emergency Services Directory",
    "med_title": "Medical & Ambulance",
    "police_title": "Police & Safety",
    "women_title": "Women Distress Helpline",
    "fire_title": "Fire & Rescue",
    "disaster_title": "Disaster Support",
    "call_txt": "Call",
}

LOCAL_T = {
    "kn_IN": {
        "brand": "ಸಹಾಯ (Sahaya)",
        "ready_status": "ಸಿದ್ಧವಾಗಿದೆ",
        "hero_title": "AI ಧ್ವನಿ ಸಹಾಯಕ",
        "subtitle": "ಮಾತನಾಡಿ. ಅನುವಾದಿಸಿ. ಸುರಕ್ಷಿತವಾಗಿರಿ.",
        "sos_btn": "🚨 ತುರ್ತು SOS • ಜೋರು ಸೈರನ್ + 112 ಗೆ ಕರೆ",
        "mic_header": "ರೆಕಾರ್ಡಿಂಗ್ ಪ್ರಾರಂಭಿಸಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "placeholder": "ಅಥವಾ ತುರ್ತು ವಿವರಗಳನ್ನು ಟೈಪ್ ಮಾಡಿ...",
        "send_btn": "ಅನುವಾದಿಸಿ & ಸಹಾಯ",
        "clear_btn": "🗑️ ಅಳಿಸಿ",
        "live_trans": "ನೇರ ತುರ್ತು ಮಾಹಿತಿ",
        "notified_sub": "ತುರ್ತು ಸೇವೆಗಳಿಗೆ ತಿಳಿಸಲಾಗಿದೆ",
        "first_aid_label": "ತಕ್ಷಣದ ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "ತುರ್ತು ಸೇವೆಗಳ ಡೈರೆಕ್ಟರಿ",
        "med_title": "ವೈದ್ಯಕೀಯ ಮತ್ತು ಆಂಬ್ಯುಲೆನ್ಸ್",
        "police_title": "ಪೊಲೀಸ್ ಮತ್ತು ಸುರಕ್ಷತೆ",
        "women_title": "ಮಹಿಳಾ ಸಹಾಯವಾಣಿ",
        "fire_title": "ಅಗ್ನಿಶಾಮಕ ಮತ್ತು ರಕ್ಷಣೆ",
        "disaster_title": "ವಿಪತ್ತು ನಿರ್ವಹಣಾ ಸಹಾಯ",
        "call_txt": "ಕರೆ ಮಾಡಿ",
    },
    "hi_IN": {
        "brand": "सहाय (Sahaya)",
        "ready_status": "तैयार",
        "hero_title": "AI वॉयस असिस्टेंट",
        "subtitle": "बोलें। अनुवाद करें। सुरक्षित रहें।",
        "sos_btn": "🚨 आपातकालीन SOS • तेज सायरन + 112 डायल करें",
        "mic_header": "रिकॉर्डिंग शुरू करने के लिए टैप करें",
        "placeholder": "या आपातकालीन विवरण टाइप करें...",
        "send_btn": "अनुवाद और सहायता",
        "clear_btn": "🗑️ साफ़ करें",
        "live_trans": "लाइव आपातकालीन ट्राइएज",
        "notified_sub": "आपातकालीन सेवाएं सूचित",
        "first_aid_label": "तत्काल प्राथमिक उपचार कदम:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "आपातकालीन सेवा निर्देशिका",
        "med_title": "चिकित्सा एवं एम्बुलेंस",
        "police_title": "पुलिस एवं सुरक्षा",
        "women_title": "महिला हेल्पलाइन",
        "fire_title": "अग्निशमन एवं बचाव",
        "disaster_title": "आपदा सहायता",
        "call_txt": "कॉल करें",
    },
}

TRANSLATIONS = {c: {**BASE_T, **LOCAL_T.get(c, {})} for c in LANGUAGES}

EMERGENCY_DIRECTORY = [
    {"key": "med_title", "icon": "🚑", "number": "108"},
    {"key": "police_title", "icon": "👮", "number": "112"},
    {"key": "women_title", "icon": "🛡️", "number": "1091"},
    {"key": "fire_title", "icon": "🚒", "number": "101"},
    {"key": "disaster_title", "icon": "🌊", "number": "1078"},
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}

CONDITION_RULES = [
    ("snakebite", {"snake", "snakes", "snakebite", "bitten", "bite", "haavu", "saanp", "paambu", "paamu"}, ["ಹಾವು", "साँप", "सांप", "பாம்பு", "పాము"]),
    ("burns", {"fire", "burn", "burns", "burnt", "cylinder", "blast", "benki", "aag"}, ["ಬೆಂಕಿ", "सुट्टा", "आग", "जला", "நெருப்பு", "మంట"]),
    ("cardiac", {"chest", "heart", "breathe", "breathing", "cardiac", "stroke", "attack", "ede", "edeno", "edenovu", "usiru", "seene", "saans"}, ["ಎದೆ", "ಉಸಿರ", "सीने", "छाती", "सांस", "दिल"]),
    ("collapse_fall", {"collapse", "collapsed", "fell", "fall", "debris", "trapped", "rubble", "building", "kattada", "bidda", "gira"}, ["ಕಟ್ಟಡ", "ಬಿದ್ದ", "इमारत", "ढह", "गिर"]),
    ("accident_trauma", {"accident", "crash", "bike", "car", "hit", "bleeding", "bleed", "blood", "wound", "injury", "apaghata", "hadsa", "khoon"}, ["ಅಪಘಾತ", "ರಕ್ತ", "ಗಾಯ", "दुर्घटना", "हादसा", "खून"]),
]

FALLBACKS = {
    "collapse_fall": {
        "en_IN": ("Building collapse / severe fall emergency! Dial 108 immediately.", [
            "Do NOT move the person (high risk of neck or spinal fracture).",
            "Keep the head, neck, and back completely straight and supported.",
            "Press firmly with a clean cloth on any open bleeding areas.",
            "Keep dust away from mouth and nose and reassure the person."
        ]),
        "hi_IN": ("इमारत ढहने या ऊंचाई से गिरने की आपात स्थिति! तुरंत 108 और 112 पर कॉल करें।", [
            "मरीज को बिल्कुल न हिलाएं, रीढ़ या गर्दन में गंभीर चोट हो सकती है।",
            "सिर, गर्दन और पीठ को सीधा और स्थिर रखें।",
            "खून बह रहा हो तो साफ कपड़े से सीधा और मजबूत दबाव डालें।",
            "बचाव दल आने तक मरीज से बात करते रहें और ढांढस बंधाएं।"
        ]),
        "kn_IN": ("ಕಟ್ಟಡ ಕುಸಿತ / ಎತ್ತರದಿಂದ ಬಿದ್ದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ರೋಗಿಯನ್ನು ಎತ್ತಬೇಡಿ ಅಥವಾ ಎಳೆಯಬೇಡಿ (ಬೆನ್ನುಮೂಳೆಯ ಗಾಯದ ಅಪಾಯ).",
            "ತಲೆ, ಕುತ್ತಿಗೆ ಮತ್ತು ಬೆನ್ನನ್ನು ಅಲುಗಾಡಿಸದೆ ನೇರವಾಗಿ ಇರಿಸಿ.",
            "ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ಗಟ್ಟಿಯಾಗಿ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.",
            "ಮುಖಕ್ಕೆ ಧೂಳು ಬರದಂತೆ ನೋಡಿಕೊಳ್ಳಿ ಮತ್ತು ರಕ್ಷಕರು ಬರುವವರೆಗೆ ಧೈರ್ಯ ಹೇಳುತ್ತಿರಿ."
        ]),
    },
    "accident_trauma": {
        "en_IN": ("Accident and severe injury emergency! Calling 108 ambulance right away.", [
            "Apply direct, firm pressure on bleeding wounds using clean cloth.",
            "Do not move the victim if spinal or neck injury is suspected.",
            "Do not offer water or food; keep them warm and responsive.",
            "If unresponsive and breathing stops, start CPR compressions at once."
        ]),
        "hi_IN": ("दुर्घटना और भारी रक्तस्राव की आपात स्थिति! तुरंत 108 पर कॉल करें।", [
            "साफ कपड़े से घाव पर सीधा और मजबूत दबाव डालें। कपड़ा भीग जाए तो हटाएं नहीं।",
            "गर्दन या पीठ में चोट की आशंका हो तो व्यक्ति को न हिलाएं।",
            "खाने या पीने को कुछ न दें। उन्हें गर्म रखें और बात करते रहें।",
            "यदि सांस न ले रहा हो तो तुरंत सीपीआर शुरू करें।"
        ]),
        "kn_IN": ("ಅಪಘಾತ / ತೀವ್ರ ರಕ್ತಸ್ರಾವದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ಸ್ವಚ್ಛ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರವಾಗಿ ಗಟ್ಟಿಯಾಗಿ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.",
            "ಕುತ್ತಿಗೆ ಅಥವಾ ಬೆನ್ನಿಗೆ ಪೆಟ್ಟಾಗಿರುವ ಸಂಶಯವಿದ್ದರೆ ವ್ಯಕ್ತಿಯನ್ನು ಅಲುಗಾಡಿಸಬೇಡಿ.",
            "ತಿನ್ನಲು ಅಥವಾ ಕುಡಿಯಲು ಏನನ್ನೂ ಕೊಡಬೇಡಿ; ಬೆಚ್ಚಗಿರಿಸಿ ಮತ್ತು ಮಾತನಾಡುತ್ತಿರಿ.",
            "ಉಸಿರಾಟ ನಿಂತಿದ್ದರೆ ತಕ್ಷಣ ಸಿಪಿಆರ್ ಪ್ರಾರಂಭಿಸಿ."
        ]),
    },
    "general_medical": {
        "en_IN": ("Medical emergency! Calling 108 ambulance immediately.", [
            "Keep the patient resting comfortably and verify they can breathe.",
            "Loosen restrictive clothing around collar and chest.",
            "If unconscious without normal breathing, begin chest compressions.",
            "Stay beside the victim until medical personnel arrive."
        ]),
        "hi_IN": ("चिकित्सा आपात स्थिति! शांत रहें और तुरंत 108 पर कॉल करें।", [
            "लगातार देखें कि व्यक्ति होश में है और सांस ले रहा है।",
            "उन्हें आराम की स्थिति में रखें और ढांढस बंधाएं।",
            "यदि वह बेहोश हो और सांस न ले रहा हो तो सीपीआर शुरू करें।",
            "108 एम्बुलेंस आने तक उनके साथ रहें।"
        ]),
        "kn_IN": ("ವೈದ್ಯಕೀಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", [
            "ವ್ಯಕ್ತಿಗೆ ಪ್ರಜ್ಞೆ ಮತ್ತು ಉಸಿರಾಟ ಇದೆಯೇ ಎಂದು ನಿರಂತರವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
            "ಆರಾಮದಾಯಕ ಭಂಗಿಯಲ್ಲಿ ವಿಶ್ರಾಂತಿ ನೀಡಿ ಮತ್ತು ಧೈರ್ಯ ಹೇಳಿ.",
            "ಪ್ರಜ್ಞೆ ಇಲ್ಲದೆ ಸಾಮಾನ್ಯ ಉಸಿರಾಟ ಇಲ್ಲದಿದ್ದರೆ ಸಿಪಿಆರ್ ಪ್ರಾರಂಭಿಸಿ.",
            "108 ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ಜೊತೆಯಲ್ಲೇ ಇರಿ."
        ]),
    }
}

def get_env(name, default=""):
    return os.getenv(name, default).strip()

def identify_condition(text: str) -> str:
    t = (text or "").lower()
    tokens = set(re.findall(r"[a-z]+", t))
    for cond, ascii_kw, native_kw in CONDITION_RULES:
        if any(k in t for k in native_kw):
            return cond
        for k in ascii_kw:
            if (" " in k and k in t) or (" " not in k and k in tokens):
                return cond
    return "general_medical"

def script_lang(text: str):
    counts = {}
    for ch in (text or ""):
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None
    best = max(counts, key=counts.get)
    if counts[best] < 2:
        return None
    if best == "hi_IN":
        mr = sum(text.count(m) for m in MR_MARKERS)
        hi = sum(text.count(m) for m in HI_MARKERS)
        return "mr_IN" if mr > hi else "hi_IN"
    return best

def autonomous_spoken_language_detector(transcript: str, w_lang: str) -> str:
    t = (transcript or "").strip()
    if not t:
        return "en_IN"

    s_lang = script_lang(t)
    if s_lang in LANGUAGES:
        return s_lang

    words = re.findall(r"[a-z]+", t.lower())
    wset = set(words)
    en_matches = len(wset & EN_COMMON)

    indic_scores = {c: len(wset & kw) for c, kw in ROMAN_LANG_KEYWORDS.items()}
    best_indic = max(indic_scores, key=indic_scores.get) if indic_scores else None
    best_score = indic_scores[best_indic] if best_indic else 0

    if best_score >= 1 and best_score >= en_matches:
        return best_indic

    if en_matches > 0 or w_lang == "en_IN":
        return "en_IN"

    if w_lang in LANGUAGES:
        return w_lang

    return "en_IN"

def fast_preprocess(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return None, "no_audio"
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception:
        return audio_path, None

    data = data.mean(axis=1)
    if len(data) < sr * 0.2:
        return None, "too_short"

    data = data[: sr * MAX_SECONDS]
    peak = float(np.max(np.abs(data)))
    if peak < 1e-4:
        return None, "silent"

    sos = butter(2, 80.0, btype="highpass", fs=sr, output="sos")
    data = sosfilt(sos, data).astype(np.float32)

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out, (data / peak * 0.95), sr, subtype="PCM_16")
    return out, None

def groq_whisper(audio_path):
    key = get_env("GROQ_API_KEY")
    if not key:
        return "", None
    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "temperature": "0",
    }
    try:
        with open(audio_path, "rb") as f:
            r = HTTP_SESSION.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data=data,
                timeout=2.2,
            )
        if r.status_code == 200:
            j = r.json()
            lang_name = str(j.get("language", "")).strip().lower()
            detected = WHISPER_NAME_TO_CODE.get(lang_name)
            return str(j.get("text", "")).strip(), detected
    except Exception:
        pass
    return "", None

def run_groq_triage(transcript: str, lang_code: str):
    key = get_env("GROQ_API_KEY")
    target_script = SCRIPT_NAMES.get(lang_code, "English")

    system_prompt = f"""You are Sahaya, an Indian emergency response triage assistant.
MANDATORY LANGUAGE RULE:
The user spoke in {lang_code} ({target_script}).
You MUST write both 'reply' and all 'steps' in {lang_code} ({target_script}).
If the user spoke in English, respond strictly in clear English.
NEVER answer in Hindi or another language if the user spoke in English.

Return JSON ONLY:
{{
  "category": "medical|police|assault|fire|disaster",
  "urgency": "critical|urgent",
  "reply": "Immediate calm reassurance in {target_script}",
  "steps": ["Step 1 in {target_script}", "Step 2 in {target_script}", "Step 3 in {target_script}"]
}}"""

    if key:
        try:
            body = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"EMERGENCY: {transcript}"},
                ],
                "temperature": 0.05,
                "max_tokens": 300,
                "response_format": {"type": "json_object"},
            }
            r = HTTP_SESSION.post(
                GROQ_CHAT_URL,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                json=body,
                timeout=1.8,
            )
            if r.status_code == 200:
                data = json.loads(r.json()["choices"][0]["message"]["content"])
                reply = str(data.get("reply", "")).strip()
                steps = [str(x).strip() for x in data.get("steps", []) if str(x).strip()]
                if reply and steps:
                    return {
                        "category": str(data.get("category", "medical")).lower(),
                        "urgency": str(data.get("urgency", "critical")).lower(),
                        "reply": reply,
                        "steps": steps,
                        "lang_code": lang_code,
                    }
        except Exception:
            pass

    cond = identify_condition(transcript)
    cat_data = FALLBACKS.get(cond, FALLBACKS["general_medical"])
    reply, steps = cat_data.get(lang_code, cat_data.get("en_IN", list(cat_data.values())[0]))
    return {
        "category": "medical",
        "urgency": "critical",
        "reply": reply,
        "steps": steps,
        "lang_code": lang_code,
    }

def fast_tts(text: str, lang_code: str):
    if lang_code not in TIMBRE_LANGS:
        return None
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return None

    payload = {
        "text": text[:350],
        "voice": get_env("TIMBRE_VOICE") or "Shivani",
        "model": get_env("TIMBRE_MODEL") or "timbre-v3.0",
        "language": lang_code.replace("_", "-"),
        "speed": 1.1,
        "audio_config": {
            "sample_rate": 24000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    try:
        r = HTTP_SESSION.post(
            TTS_URL,
            headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=2.0,
        )
        if r.status_code == 200 and r.content:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(r.content)
            tmp.close()
            return tmp.name
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# High-Contrast UI HTML Renderers
# ---------------------------------------------------------------------------
def header_html(t):
    return f"""
    <div class="app-top-bar">
      <div class="brand-left">
        <div class="brand-wave-icon" aria-label="Soundwave">
          <span></span><span></span><span></span><span></span><span></span>
        </div>
        <span class="brand-name">{html.escape(t['brand'])}</span>
      </div>
      <div class="connected-pill">
        <span class="conn-dot"></span>
        <span class="conn-text-label">{html.escape(t['ready_status'])}</span>
      </div>
      <button type="button" class="gear-icon-btn" onclick="window.openSettingsModal()" title="Settings">
        <svg class="gear-svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
      </button>
    </div>
    """

def custom_mic_card_html(t):
    return f"""
    <div class="assistant-hero-card">
      <h1 class="hero-title">{html.escape(t['hero_title'])}</h1>
      <p class="hero-subtitle">{html.escape(t['subtitle'])}</p>

      <div class="mic-stage">
        <div class="mic-ring-outer">
          <div class="mic-ring-inner">
            <button type="button" class="blue-mic-btn" id="customMicBtn" onclick="window.toggleSahayaMic()">
              <svg id="svgMicIcon" width="36" height="36" viewBox="0 0 24 24" fill="white">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
              <svg id="svgStopIcon" width="28" height="28" viewBox="0 0 24 24" fill="white" style="display:none;">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <div class="mic-prompt-main" id="micPromptMain">{html.escape(t['mic_header'])}</div>

      <div class="lang-pill-wrap">
        <div class="autonomous-detect-badge">
          ⚡ <span>{html.escape(t['auto_detect_badge'])}</span>
        </div>
      </div>

      <div class="shield-banner">
        <div class="shield-text">🛡️ {html.escape(t['shield_text'])}</div>
      </div>
    </div>
    """

def result_card(service, steps, urgency, rt, reply_text, caller_text="", detected_speech=""):
    steps_html = "".join(f"<li class='step-item'>{html.escape(s)}</li>" for s in steps)
    caller_bubble = ""
    if caller_text:
        caller_bubble = f"""
        <div class="trans-bubble-blue">
          <span class="bubble-vol">🗣️</span>
          <div class="bubble-txt">{html.escape(caller_text)} <span class="detected-lang-pill">({detected_speech})</span></div>
        </div>
        """
    return f"""
    <div class="accident-result-card">
      <div class="accident-alert-top">
        <div class="alert-triangle-icon">⚠️</div>
        <h2 class="accident-main-title">{html.escape(rt.get(service['key'], 'Emergency Alert'))}</h2>
        <p class="accident-sub-note">{html.escape(rt.get('notified_sub', 'Emergency Services Notified'))} • {html.escape(urgency.upper())}</p>
      </div>

      <div class="live-trans-surface">
        <div class="live-trans-header-row">
          <div class="live-trans-header">
            <span>🚨 {html.escape(rt.get('live_trans', 'Emergency Response'))}</span>
          </div>
          <button type="button" class="inline-clear-btn" onclick="window.triggerClearEmergency()">
            {html.escape(rt.get('clear_btn', 'Clear'))}
          </button>
        </div>
        {caller_bubble}
        <div class="trans-bubble-gray">
          <span class="bubble-vol">🔊</span>
          <div class="bubble-txt">{html.escape(reply_text)}</div>
        </div>

        <div class="first-aid-section">
          <div class="first-aid-title">🩺 {html.escape(rt.get('first_aid_label', 'Immediate Steps:'))}</div>
          <ol class="step-list">{steps_html}</ol>
        </div>

        <a class="dial-now-btn" href="tel:{service['number']}">
          📞 {html.escape(rt.get('call_btn', 'Call {number}').format(number=service['number']))}
        </a>
      </div>
    </div>
    """

def directory_html(t):
    cards = []
    for srv in EMERGENCY_DIRECTORY:
        cards.append(f"""
        <div class="directory-card">
          <div class="directory-info">
            <span class="directory-icon">{srv['icon']}</span>
            <div>
              <div class="directory-name">{html.escape(t.get(srv['key'], srv['key']))}</div>
              <div class="directory-num">{srv['number']}</div>
            </div>
          </div>
          <a class="call-badge" href="tel:{srv['number']}">{html.escape(t.get('call_txt', 'Call'))}</a>
        </div>""")
    return f"""
    <div class="directory-section">
      <h4 class="directory-heading">📁 {html.escape(t['directory_header'])}</h4>
      <div class="directory-grid">{''.join(cards)}</div>
    </div>"""

# ---------------------------------------------------------------------------
# Execution Workflows
# ---------------------------------------------------------------------------
def process_pipeline(transcript: str, detected_lang: str):
    triage = run_groq_triage(transcript, detected_lang)
    svc = EMERGENCY_DIRECTORY[CATEGORY_INDEX.get(triage["category"], 0)]
    rt = TRANSLATIONS.get(detected_lang, TRANSLATIONS["en_IN"])

    card = result_card(svc, triage["steps"], triage["urgency"], rt, triage["reply"], caller_text=transcript, detected_speech=detected_lang)
    yield gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value=card, visible=True)

    narration = f"{triage['reply']} " + " ".join(triage["steps"][:2])
    audio_path = fast_tts(narration, detected_lang)
    yield gr.update(value="", visible=False), gr.update(value=audio_path, visible=bool(audio_path)), gr.update(value=card, visible=True)

def on_b64_audio(b64_data):
    if not b64_data or "," not in b64_data:
        yield gr.update(value="⚠️ Tap microphone and speak your emergency.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return
    try:
        raw_bytes = base64.b64decode(b64_data.split(",", 1)[1])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(tmp, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        yield gr.update(value=f"⚠️ Audio error: {e}", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return

    clean, err = fast_preprocess(tmp)
    if err:
        yield gr.update(value="⚠️ Could not hear audio clearly. Please try again.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return

    transcript, w_lang = groq_whisper(clean)
    if not transcript:
        yield gr.update(value="⚠️ No speech recognized. Please speak clearly.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return

    detected_lang = autonomous_spoken_language_detector(transcript, w_lang)
    yield from process_pipeline(transcript, detected_lang)

def on_text_submit(text_in):
    cleaned = (text_in or "").strip()
    if not cleaned:
        yield gr.update(value="⚠️ Please enter emergency details.", visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False)
        return
    detected_lang = autonomous_spoken_language_detector(cleaned, None)
    yield from process_pipeline(cleaned, detected_lang)

def on_clear():
    return "", gr.update(value="", visible=False), gr.update(value=None, visible=False), gr.update(value="", visible=False), ""

def set_app_ui_language(lang_code: str):
    target = lang_code if lang_code in LANGUAGES else "en_IN"
    t = TRANSLATIONS.get(target, TRANSLATIONS["en_IN"])
    return (
        header_html(t),
        t["sos_btn"],
        custom_mic_card_html(t),
        gr.update(placeholder=t["placeholder"]),
        t["send_btn"],
        t["clear_btn"],
        directory_html(t),
    )

# ---------------------------------------------------------------------------
# CSS Overhaul (High Contrast, Solid Backgrounds, Absolute Readability)
# ---------------------------------------------------------------------------
CSS = """
:root, body, .gradio-container {
    --bg-main: #F4F6F9;
    --card-bg: #FFFFFF;
    --card-border: #CBD5E1;
    --text-primary: #0F172A;
    --topbar-bg: #FFFFFF;
}

body.dark-mode, .dark-mode .gradio-container {
    --bg-main: #111827 !important;
    --card-bg: #1F2937 !important;
    --card-border: #374151 !important;
    --text-primary: #FFFFFF !important;
    --topbar-bg: #1F2937 !important;
}

body, .gradio-container {
    background: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    max-width: 500px !important;
    margin: 0 auto !important;
    padding: 6px 10px 14px 10px !important;
}

footer { display: none !important; }
.js-bridge-hidden { display: none !important; }

/* Animated Sound Wave */
.brand-wave-icon {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 22px;
}
.brand-wave-icon span {
    display: inline-block;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #F97316 0%, #2563EB 100%);
    border-radius: 4px;
    animation: waveBounce 1.2s ease-in-out infinite alternate;
}
.brand-wave-icon span:nth-child(1) { height: 8px; animation-delay: 0.1s; }
.brand-wave-icon span:nth-child(2) { height: 18px; animation-delay: 0.3s; }
.brand-wave-icon span:nth-child(3) { height: 12px; animation-delay: 0.15s; }
.brand-wave-icon span:nth-child(4) { height: 20px; animation-delay: 0.4s; }
.brand-wave-icon span:nth-child(5) { height: 10px; animation-delay: 0.2s; }

@keyframes waveBounce {
    0% { transform: scaleY(0.35); }
    100% { transform: scaleY(1.0); }
}

/* App Bar */
.app-top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--topbar-bg) !important;
    border-radius: 16px;
    border: 1.5px solid var(--card-border);
    margin-bottom: 8px;
}
.brand-name { font-size: 22px; font-weight: 800; color: #2563EB !important; }

.connected-pill {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: #DCFCE7 !important;
    border: 1.5px solid #86EFAC !important;
    padding: 4px 12px !important;
    border-radius: 999px !important;
}
.conn-dot {
    width: 8px; height: 8px;
    background: #16A34A !important;
    border-radius: 50%;
    display: inline-block;
}
.conn-text-label {
    font-size: 13px !important;
    font-weight: 800 !important;
    color: #166534 !important;
}

.gear-icon-btn {
    background: #F1F5F9 !important;
    border: 1.5px solid #64748B !important;
    padding: 6px;
    border-radius: 50%;
    cursor: pointer;
}

.sos-panic-btn {
    width: 100% !important;
    background: #DC2626 !important;
    color: #FFFFFF !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
}

/* Mic Stage */
.assistant-hero-card {
    background: var(--card-bg) !important;
    border-radius: 20px;
    padding: 18px 16px;
    text-align: center;
    border: 1.5px solid var(--card-border);
    margin-bottom: 8px;
}
.hero-title {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
.hero-subtitle {
    font-size: 14px !important;
    color: #2563EB !important;
    font-weight: 700 !important;
    margin: 3px 0 12px 0 !important;
}

.mic-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 8px 0;
}
.mic-ring-outer {
    width: 120px; height: 120px; border-radius: 50%;
    background: rgba(37, 99, 235, 0.12);
    display: flex; align-items: center; justify-content: center;
}
.mic-ring-inner {
    width: 90px; height: 90px; border-radius: 50%;
    background: rgba(37, 99, 235, 0.2);
    display: flex; align-items: center; justify-content: center;
}
.blue-mic-btn {
    width: 68px; height: 68px; border-radius: 50%;
    background: #2563EB !important; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}
.blue-mic-btn.recording-active { background: #DC2626 !important; }

.mic-prompt-main {
    font-size: 15px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin-top: 10px !important;
}

.lang-pill-wrap { margin-top: 10px; display: flex; justify-content: center; }
.autonomous-detect-badge {
    background: #E2E8F0 !important;
    border: 1.5px solid #64748B !important;
    padding: 6px 14px !important;
    border-radius: 999px !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
}
.dark-mode .autonomous-detect-badge {
    background: #374151 !important;
    border-color: #6B7280 !important;
    color: #F9FAFB !important;
}

.shield-banner {
    background: rgba(22, 163, 74, 0.15) !important;
    border: 1.5px solid rgba(22, 163, 74, 0.4) !important;
    border-radius: 10px;
    padding: 8px 12px;
    margin-top: 12px;
}
.shield-text {
    font-size: 12px !important;
    font-weight: 800 !important;
    color: #166534 !important;
}
.dark-mode .shield-text { color: #86EFAC !important; }

/* Text Submission Input */
.quick-text-row {
    background: var(--card-bg) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    display: flex !important;
    gap: 6px !important;
    margin-bottom: 8px !important;
    border: 1.5px solid var(--card-border) !important;
}
.quick-text-row textarea,
.quick-text-row input {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #94A3B8 !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
.dark-mode .quick-text-row textarea,
.dark-mode .quick-text-row input {
    background: #111827 !important;
    color: #F9FAFB !important;
    border-color: #6B7280 !important;
}
.send-text-btn { background: #2563EB !important; color: #FFFFFF !important; font-weight: 800 !important; border-radius: 10px !important; }
.clear-text-btn { background: #E2E8F0 !important; color: #0F172A !important; font-weight: 700 !important; border-radius: 10px !important; border: 1.5px solid #94A3B8 !important; }
.dark-mode .clear-text-btn { background: #374151 !important; color: #F9FAFB !important; border-color: #6B7280 !important; }

/* Directory Grid */
.directory-section { margin-top: 6px; }
.directory-heading {
    font-size: 14px !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 0 8px 2px !important;
}
.directory-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}
.directory-card {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--card-border) !important;
    border-radius: 12px;
    padding: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.directory-name {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
.directory-num {
    font-size: 13px !important;
    font-weight: 800 !important;
    color: #DC2626 !important;
}
.call-badge {
    background: #DBEAFE !important;
    color: #1D4ED8 !important;
    border: 1px solid #93C5FD !important;
    padding: 4px 8px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 12px;
}

/* Result Card & Bubbles (High Contrast Fix) */
.accident-result-card {
    background: var(--card-bg) !important;
    border-radius: 18px;
    border: 2px solid #EF4444 !important;
    overflow: hidden;
    margin-bottom: 8px;
}
.accident-alert-top { background: #FEE2E2 !important; padding: 12px; text-align: center; }
.dark-mode .accident-alert-top { background: #7F1D1D !important; }
.accident-main-title { font-size: 18px !important; font-weight: 800 !important; color: #B91C1C !important; margin: 0; }
.dark-mode .accident-main-title { color: #FECACA !important; }
.accident-sub-note { font-size: 12px !important; font-weight: 700 !important; color: #7F1D1D !important; margin: 4px 0 0 0; }
.dark-mode .accident-sub-note { color: #FCA5A5 !important; }

.live-trans-surface {
    padding: 14px;
    background: var(--card-bg) !important;
    color: var(--text-primary) !important;
}
.live-trans-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.live-trans-header { font-weight: 800 !important; color: var(--text-primary) !important; font-size: 13px; }
.inline-clear-btn { background: #FEE2E2 !important; color: #DC2626 !important; border: 1.5px solid #F87171 !important; border-radius: 6px; padding: 3px 8px; cursor: pointer; font-weight: 700; }

.trans-bubble-blue {
    background: #EFF6FF !important;
    border: 1.5px solid #3B82F6 !important;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.dark-mode .trans-bubble-blue {
    background: #1E293B !important;
    border-color: #60A5FA !important;
}
.trans-bubble-blue .bubble-txt {
    color: #1E3A8A !important;
    font-weight: 800 !important;
    font-size: 14px !important;
}
.dark-mode .trans-bubble-blue .bubble-txt {
    color: #BFDBFE !important;
}

.trans-bubble-gray {
    background: #F1F5F9 !important;
    border: 1.5px solid #64748B !important;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.dark-mode .trans-bubble-gray {
    background: #374151 !important;
    border-color: #9CA3AF !important;
}
.trans-bubble-gray .bubble-txt {
    color: #0F172A !important;
    font-weight: 800 !important;
    font-size: 14px !important;
}
.dark-mode .trans-bubble-gray .bubble-txt {
    color: #F9FAFB !important;
}

.detected-lang-pill {
    font-size: 11px;
    font-weight: 800;
    opacity: 0.85;
}

.first-aid-section { color: var(--text-primary) !important; }
.first-aid-title { font-size: 14px !important; font-weight: 800 !important; color: var(--text-primary) !important; margin-bottom: 6px; }

.step-list {
    margin: 0 0 12px 0;
    padding-left: 20px;
    font-size: 13px;
    color: var(--text-primary) !important;
}
.step-item {
    margin-bottom: 6px;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

.dial-now-btn {
    display: block;
    text-align: center;
    background: #16A34A !important;
    color: #FFFFFF !important;
    padding: 10px;
    border-radius: 10px;
    text-decoration: none !important;
    font-weight: 800;
    font-size: 14px;
}

/* Modal */
.settings-modal-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(4px);
    display: none; justify-content: center; align-items: center; z-index: 9999;
}
.settings-modal-content {
    background: #FFFFFF !important; color: #0F172A !important; padding: 20px;
    border-radius: 18px; width: 90%; max-width: 380px;
    border: 2px solid #CBD5E1;
}
.dark-mode .settings-modal-content { background: #1F2937 !important; color: #FFFFFF !important; border-color: #4B5563; }
.settings-modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.settings-close-btn { border: none; background: #E2E8F0; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; font-weight: 800; }
.dark-mode .settings-close-btn { background: #374151; color: #FFFFFF; }
.setting-item { margin-bottom: 14px; }
.setting-item label { font-size: 12px; font-weight: 800; color: #475569; display: block; margin-bottom: 6px; }
.dark-mode .setting-item label { color: #D1D5DB !important; }
.setting-item select { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1.5px solid #CBD5E1; font-weight: 700; color: #0F172A; background: #FFF; }
.dark-mode .setting-item select { background: #111827 !important; color: #FFFFFF !important; border-color: #4B5563 !important; }
"""

INIT_JS = """() => {
    window._sahayaRecording = false;
    window._audioCtx = null;
    window._mediaStream = null;
    window._scriptNode = null;
    window._pcmChunks = [];

    window.pushToGradio = function(boxId, btnId, val) {
        const el = document.querySelector('#' + boxId + ' textarea, #' + boxId + ' input');
        if (!el) return;
        const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
        if (setter) setter.call(el, val); else el.value = val;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        setTimeout(function() {
            const btn = document.getElementById(btnId);
            if (btn) btn.click();
        }, 30);
    };

    window.triggerClearEmergency = function() {
        const btn = document.getElementById('visible-clear-btn');
        if (btn) btn.click();
    };

    window.encodeWAVBase64 = function(buffers, inputSampleRate) {
        let totalLength = 0;
        for (let i = 0; i < buffers.length; i++) {
            totalLength += buffers[i].length;
        }
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (let i = 0; i < buffers.length; i++) {
            merged.set(buffers[i], offset);
            offset += buffers[i].length;
        }
        const targetRate = 16000;
        const ratio = inputSampleRate / targetRate;
        const samples = new Float32Array(Math.round(merged.length / ratio));
        for (let i = 0; i < samples.length; i++) {
            samples[i] = merged[Math.min(merged.length - 1, Math.floor(i * ratio))];
        }
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const writeString = function(off, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(off + i, str.charCodeAt(i));
            }
        };
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, targetRate, true);
        view.setUint32(28, targetRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let idx = 44;
        for (let i = 0; i < samples.length; i++, idx += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(idx, s < 0 ? s * 32768 : s * 32767, true);
        }

        const bytes = new Uint8Array(buffer);
        let binary = '';
        const chunkSize = 32768;
        for (let i = 0; i < bytes.length; i += chunkSize) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
        }
        return 'data:audio/wav;base64,' + btoa(binary);
    };

    window.toggleSahayaMic = async function() {
        const btn = document.getElementById('customMicBtn');
        const micSvg = document.getElementById('svgMicIcon');
        const stopSvg = document.getElementById('svgStopIcon');
        const promptMain = document.getElementById('micPromptMain');

        if (window._sahayaRecording) {
            window._sahayaRecording = false;
            try { if (window._scriptNode) window._scriptNode.disconnect(); } catch(e) {}
            try { if (window._mediaStream) window._mediaStream.getTracks().forEach(function(t) { t.stop(); }); } catch(e) {}
            try { if (window._audioCtx) window._audioCtx.close(); } catch(e) {}

            if (btn) btn.classList.remove('recording-active');
            if (micSvg) micSvg.style.display = 'block';
            if (stopSvg) stopSvg.style.display = 'none';
            if (promptMain) promptMain.textContent = 'Processing emergency speech...';

            if (window._pcmChunks.length > 0) {
                const b64 = window.encodeWAVBase64(window._pcmChunks, 44100);
                window._pcmChunks = [];
                window.pushToGradio('hidden-b64-audio', 'hidden-audio-submit-btn', b64);
            }
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            window._mediaStream = stream;
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioCtx();
            window._audioCtx = ctx;
            const source = ctx.createMediaStreamSource(stream);
            const processor = ctx.createScriptProcessor(4096, 1, 1);
            window._scriptNode = processor;
            window._pcmChunks = [];
            window._sahayaRecording = true;

            processor.onaudioprocess = function(e) {
                if (!window._sahayaRecording) return;
                window._pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
            };

            source.connect(processor);
            processor.connect(ctx.destination);

            if (btn) btn.classList.add('recording-active');
            if (micSvg) micSvg.style.display = 'none';
            if (stopSvg) stopSvg.style.display = 'block';
            if (promptMain) promptMain.textContent = 'Recording... (Tap button to stop)';
        } catch (err) {
            alert('Please allow microphone permissions to record.');
        }
    };

    window.openSettingsModal = function() {
        let modal = document.getElementById('sahayaSettingsModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'sahayaSettingsModal';
            modal.className = 'settings-modal-overlay';
            modal.innerHTML = `
                <div class="settings-modal-content">
                    <div class="settings-modal-header">
                        <h3 style="margin:0;">⚙️ App Settings</h3>
                        <button class="settings-close-btn" onclick="window.closeSettingsModal()">&times;</button>
                    </div>
                    <div class="setting-item">
                        <label>🌐 Preferred Language (Translates Entire App)</label>
                        <select id="modalLangSelect" onchange="window.syncLanguageSelection(this.value)">
                            <option value="en_IN">English (India)</option>
                            <option value="kn_IN">ಕನ್ನಡ (Kannada)</option>
                            <option value="hi_IN">हिन्दी (Hindi)</option>
                            <option value="ta_IN">தமிழ் (Tamil)</option>
                            <option value="te_IN">తెలుగు (Telugu)</option>
                            <option value="ml_IN">മലയാളം (Malayalam)</option>
                            <option value="mr_IN">मराठी (Marathi)</option>
                            <option value="gu_IN">ગુજરાતી (Gujarati)</option>
                            <option value="bn_IN">বাংলা (Bengali)</option>
                            <option value="pa_IN">ਪੰਜਾਬੀ (Punjabi)</option>
                            <option value="or_IN">ଓଡ଼ିଆ (Odia)</option>
                            <option value="mni_IN">ꯃꯤꯇꯩꯂꯣꯟ (Manipuri)</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>🎨 Theme Mode</label>
                        <select onchange="document.body.classList.toggle('dark-mode', this.value === 'dark')">
                            <option value="light">☀️ Light Theme</option>
                            <option value="dark">🌙 Dark Theme</option>
                        </select>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }
        modal.style.display = 'flex';
    };

    window.closeSettingsModal = function() {
        const modal = document.getElementById('sahayaSettingsModal');
        if (modal) modal.style.display = 'none';
    };

    window.syncLanguageSelection = function(langCode) {
        window.pushToGradio('hidden-lang-box', 'hidden-lang-submit-btn', langCode);
        window.closeSettingsModal();
    };
}"""

# ---------------------------------------------------------------------------
# Gradio App Definition
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(title="Sahaya Emergency") as demo:
    header = gr.HTML(header_html(t0))

    sos_btn = gr.Button(t0["sos_btn"], elem_classes=["sos-panic-btn"])
    sos_btn.click(
        fn=None,
        js="""() => { window.open('tel:112', '_self'); }"""
    )

    mic_card = gr.HTML(custom_mic_card_html(t0))

    with gr.Row(elem_classes=["quick-text-row"]):
        text_in = gr.Textbox(placeholder=t0["placeholder"], show_label=False, scale=5, container=False)
        send_btn = gr.Button(t0["send_btn"], elem_classes=["send-text-btn"], scale=2)
        clear_btn = gr.Button(t0["clear_btn"], elem_id="visible-clear-btn", elem_classes=["clear-text-btn"], scale=1)

    status = gr.Markdown("", visible=False)
    reply_audio = gr.Audio(label="🔊 Voice Guidance", type="filepath", autoplay=True, interactive=False, visible=False)
    card_out = gr.HTML("", visible=False)

    dir_html = gr.HTML(directory_html(t0))

    with gr.Row(elem_classes=["js-bridge-hidden"]):
        lang_state = gr.Textbox(value="en_IN", elem_id="hidden-lang-box")
        lang_btn = gr.Button("ApplyLang", elem_id="hidden-lang-submit-btn")
        b64_audio_in = gr.Textbox(value="", elem_id="hidden-b64-audio")
        b64_audio_btn = gr.Button("SubmitAudio", elem_id="hidden-audio-submit-btn")

    b64_audio_btn.click(
        on_b64_audio,
        inputs=[b64_audio_in],
        outputs=[status, reply_audio, card_out],
    )

    send_btn.click(
        on_text_submit,
        inputs=[text_in],
        outputs=[status, reply_audio, card_out],
    )
    text_in.submit(
        on_text_submit,
        inputs=[text_in],
        outputs=[status, reply_audio, card_out],
    )

    clear_btn.click(
        on_clear,
        inputs=None,
        outputs=[text_in, status, reply_audio, card_out, b64_audio_in],
    )

    lang_btn.click(
        set_app_ui_language,
        inputs=lang_state,
        outputs=[header, sos_btn, mic_card, text_in, send_btn, clear_btn, dir_html],
    )

    demo.queue(default_concurrency_limit=10, max_size=50)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Base(),
        css=CSS,
        js=INIT_JS,
    )
