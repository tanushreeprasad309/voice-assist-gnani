"""
Sahaya Emergency Assistant (Gradio on Render)
Voice-first emergency triage and first aid for India across 12 languages.

Gnani AI stack:
Prisma  (speech-to-text)   -> https://api.vachana.ai/stt/v3
Timbre  (text-to-speech)   -> https://api.vachana.ai/api/v1/tts/inference
Groq (triage reasoning)    -> https://api.groq.com/openai/v1/chat/completions
"""

import html
import json
import os
import re
import tempfile
import gradio as gr
import requests

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# ---------------------------------------------------------------------------
# Languages & UI Translations (12 Indian Languages)
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

LANG_NAMES = {
    "en_IN": "English",
    "kn_IN": "Kannada (ಕನ್ನಡ)",
    "hi_IN": "Hindi (हिन्दी)",
    "ta_IN": "Tamil (தமிழ்)",
    "te_IN": "Telugu (తెలుగు)",
    "ml_IN": "Malayalam (മലയാളം)",
    "mr_IN": "Marathi (मराठी)",
    "gu_IN": "Gujarati (ગુજરાતી)",
    "bn_IN": "Bengali (বাংলা)",
    "pa_IN": "Punjabi (ਪੰਜਾਬੀ)",
    "or_IN": "Odia (ଓଡ଼ିଆ)",
    "mni_IN": "Manipuri (Meitei Mayek)",
}

TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

TRANSLATIONS = {
    "en_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe. 🛡️",
        "mic_header": "🎙️ Speak Emergency (Tap Mic to Start & Stop)",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "e.g., Road accident, severe bleeding, chest pain, burns, snake bite...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ Detected Emergency:",
        "action_header": "Recommended Action & First-Aid",
        "first_aid_label": "Immediate First-Aid Steps:",
        "call_btn": "Call {number} Immediately",
        "directory_header": "Emergency Services Directory",
        "helpline": "Helpline:",
        "audio_response": "🔊 Sahaya Voice Guidance:",
    },
    "kn_IN": {
        "title": "ಸಹಾಯ (Sahaya)",
        "subtitle": "ಸುರಕ್ಷಿತವಾಗಿರಿ. 🛡️",
        "mic_header": "🎙️ ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಮಾತನಾಡಿ (ಪ್ರಾರಂಭಿಸಲು ಮೈಕ್ ಟ್ಯಾಪ್ ಮಾಡಿ)",
        "type_header": "⌨️ ಅಥವಾ ತುರ್ತು ವಿವರಗಳನ್ನು ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ",
        "type_placeholder": "ಉದಾ. ಅಪಘಾತ, ತೀವ್ರ ರಕ್ತಸ್ರಾವ, ಎದೆನೋವು, ಸುಟ್ಟ ಗಾಯ, ಹಾವು ಕಚ್ಚಿದೆ...",
        "send": "ತುರ್ತು ಎಚ್ಚರಿಕೆ ಕಳುಹಿಸಿ",
        "detected": "🗣️ ಪತ್ತೆಯಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ:",
        "action_header": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಮತ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "first_aid_label": "ತಕ್ಷಣದ ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "ತುರ್ತು ಸಹಾಯವಾಣಿ ಡೈರೆಕ್ಟರಿ",
        "helpline": "ಸಹಾಯವಾಣಿ:",
        "audio_response": "🔊 ಸಹಾಯ ಧ್ವನಿ ಮಾರ್ಗದರ್ಶನ:",
    },
    "hi_IN": {
        "title": "सहायता (Sahaya)",
        "subtitle": "सुरक्षित रहें। 🛡️",
        "mic_header": "🎙️ आपातकालीन स्थिति बोलें (माइक टैप करें)",
        "type_header": "⌨️ या आपातकालीन विवरण टाइप करें",
        "type_placeholder": "जैसे सड़क दुर्घटना, खून बह रहा है, सीने में दर्द, जलना...",
        "send": "आपातकालीन अलर्ट भेजें",
        "detected": "🗣️ पता चली आपात स्थिति:",
        "action_header": "अनुशंसित कार्रवाई और प्राथमिक उपचार",
        "first_aid_label": "प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "आपातकालीन सेवा डायरेक्टरी",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 सहायता वॉयस मार्गदर्शन:",
    },
    "ta_IN": {
        "title": "சஹாயா (Sahaya)",
        "subtitle": "பாதுகாப்பாக இருங்கள். 🛡️",
        "mic_header": "🎙️ அவசரத்தைப் பேச மைக் தொடவும்",
        "type_header": "⌨️ அவசர விவரங்களை தட்டச்சு செய்யவும்",
        "type_placeholder": "எ.கா. விபத்து, அதிக ரத்தப்போக்கு, நெஞ்சு வலி, தீக்காயம்...",
        "send": "அவசர எச்சரிக்கை அனுப்பு",
        "detected": "🗣️ கண்டறியப்பட்ட அவசரம்:",
        "action_header": "பரிந்துரைக்கப்பட்ட முதலுதவி",
        "first_aid_label": "உடனடி முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "அவசர சேவை அடைவு",
        "helpline": "உதவி எண்:",
        "audio_response": "🔊 சஹாயா குரல் வழிகாட்டல்:",
    },
    "te_IN": {
        "title": "సహాయ (Sahaya)",
        "subtitle": "సురక్షితంగా ఉండండి. 🛡️",
        "mic_header": "🎙️ అత్యవసరం మాట్లాడటానికి మైక్ నొక్కండి",
        "type_header": "⌨️ అత్యవసర వివరాలు టైప్ చేయండి",
        "type_placeholder": "ఉదా. ప్రమాదం, రక్తస్రావం, గుండె నొప్పి, కాలిన గాయాలు...",
        "send": "అత్యవసర హెచ్చరిక పంపండి",
        "detected": "🗣️ గుర్తించబడిన అత్యవసరం:",
        "action_header": "సిఫార్సు చేయబడిన ప్రథమ చికిత్స",
        "first_aid_label": "తక్షణ ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "అత్యవసర సేవల డైరెక్టరీ",
        "helpline": "హెల్ప్‌లైన్:",
        "audio_response": "🔊 సహాయ వాయిస్ మార్గదర్శకత్వం:",
    },
    "ml_IN": {
        "title": "സഹായ (Sahaya)",
        "subtitle": "സുരക്ഷിതരായിരിക്കൂ. 🛡️",
        "mic_header": "🎙️ സംസാരിക്കാൻ മൈക്ക് ടാപ്പ് ചെയ്യുക",
        "type_header": "⌨️ വിവരങ്ങൾ ടൈപ്പ് ചെയ്യുക",
        "type_placeholder": "ഉദാ. അപകടം, രക്തസ്രാവം, നെഞ്ചുവേദന, പൊള്ളൽ...",
        "send": "അടിയന്തര മുന്നറിയിപ്പ് അയയ്ക്കുക",
        "detected": "🗣️ കണ്ടെത്തിയ അത്യാഹിതം:",
        "action_header": "ശുപാർശ ചെയ്യുന്ന പ്രഥമശുശ്രൂഷ",
        "first_aid_label": "ഉടൻ ചെയ്യേണ്ട പ്രഥമശുശ്രൂഷാ ഘട്ടങ്ങൾ:",
        "call_btn": "ഉടൻ {number}-ലേക്ക് വിളിക്കുക",
        "directory_header": "എമർജൻസി ഡയറക്ടറി",
        "helpline": "ഹെൽപ്പ്‌ലൈൻ:",
        "audio_response": "🔊 സഹായ വോയ്സ് പ്രതികരണം:",
    },
    "mr_IN": {
        "title": "साहाय्य (Sahaya)",
        "subtitle": "सुरक्षित राहा. 🛡️",
        "mic_header": "🎙️ बोलण्यासाठी माइक टॅप करा",
        "type_header": "⌨️ किंवा टाइप करा",
        "type_placeholder": "उदा. अपघात, रक्तस्त्राव, छातीत दुखणे, भाजणे...",
        "send": "आणीबाणी सूचना पाठवा",
        "detected": "🗣️ आढळलेली आणीबाणी:",
        "action_header": "शिफारस केलेले प्रथमोपचार",
        "first_aid_label": "तातडीच्या प्रथमोपचार सूचना:",
        "call_btn": "आता {number} वर कॉल करा",
        "directory_header": "आणीबाणी निर्देशिका",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 साहाय्य व्हॉइस मार्गदर्शन:",
    },
    "gu_IN": {
        "title": "સહાય (Sahaya)",
        "subtitle": "સુરક્ષિત રહો. 🛡️",
        "mic_header": "🎙️ બોલવા માટે માઇક દબાવો",
        "type_header": "⌨️ અથવા વિગતો ટાઇપ કરો",
        "type_placeholder": "દા.ત. અકસ્માત, લોહી વહેવું, છાતીમાં દુખાવો, દાઝવું...",
        "send": "કટોકટી ચેતવણી મોકલો",
        "detected": "🗣️ મળેલી કટોકટી:",
        "action_header": "પ્રાથમિક સારવાર માર્ગદર્શિકા",
        "first_aid_label": "તાત્કાલિક પ્રાથમિક સારવારનાં પગલાં:",
        "call_btn": "હમણાં {number} પર કૉલ કરો",
        "directory_header": "ઇમરજન્સી ડિરેક્ટરી",
        "helpline": "હેલ્પલાઇન:",
        "audio_response": "🔊 સહાય વૉઇસ માર્ગદર્શન:",
    },
    "bn_IN": {
        "title": "সহায় (Sahaya)",
        "subtitle": "সুরক্ষিত থাকুন। 🛡️",
        "mic_header": "🎙️ কথা বলতে মাইক ট্যাপ করুন",
        "type_header": "⌨️ জরুরি বিবরণ টাইপ করুন",
        "type_placeholder": "যেমন দুর্ঘটনা, রক্তপাত, বুকে ব্যথা, পোড়া আঘাত...",
        "send": "জরুরি সতর্কতা পাঠান",
        "detected": "🗣️ শনাক্ত জরুরি অবস্থা:",
        "action_header": "প্রাথমিক চিকিৎসার নির্দেশিকা",
        "first_aid_label": "জরুরি প্রাথমিক চিকিৎসার পদক্ষেপ:",
        "call_btn": "এখনই {number}-এ কল করুন",
        "directory_header": "জরুরি ডিরেক্টরি",
        "helpline": "হেল্পলাইন:",
        "audio_response": "🔊 সহায় ভয়েস সহায়তা:",
    },
    "pa_IN": {
        "title": "ਸਹਾਇਤਾ (Sahaya)",
        "subtitle": "ਸੁਰੱਖਿਅਤ ਰਹੋ। 🛡️",
        "mic_header": "🎙️ ਬੋਲਣ ਲਈ ਮਾਈਕ ਟੈਪ ਕਰੋ",
        "type_header": "⌨️ ਜਾਂ ਵੇਰਵੇ ਟਾਈਪ ਕਰੋ",
        "type_placeholder": "ਜਿਵੇਂ ਹਾਦਸਾ, ਖੂਨ ਵਹਿਣਾ, ਛਾਤੀ ਵਿੱਚ ਦਰਦ, ਸੜਨਾ...",
        "send": "ਐਮਰਜੈਂਸੀ ਅਲਰਟ ਭੇਜੋ",
        "detected": "🗣️ ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ:",
        "action_header": "ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼",
        "first_aid_label": "ਤੁਰੰਤ ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਕਦਮ:",
        "call_btn": "ਹੁਣੇ {number} 'ਤੇ ਕਾਲ ਕਰੋ",
        "directory_header": "ਐਮਰਜੈਂਸੀ ਡਾਇਰੈਕਟਰੀ",
        "helpline": "ਹੈਲਪਲਾਈਨ:",
        "audio_response": "🔊 ਸਹਾਇਤਾ ਵੌਇਸ ਸੇਧ:",
    },
    "or_IN": {
        "title": "ସହାୟ (Sahaya)",
        "subtitle": "ସୁରକ୍ଷିତ ରୁହନ୍ତୁ। 🛡️",
        "mic_header": "🎙️ କହିବାକୁ ମାଇକ୍ ଟ୍ୟାପ୍ କରନ୍ତୁ",
        "type_header": "⌨️ ଜରୁରୀ ବିବରଣୀ ଟାଇପ୍ କରନ୍ତୁ",
        "type_placeholder": "ଯେପରି ଦୁର୍ଘଟଣା, ରକ୍ତସ୍ରାବ, ଛାତି ଯନ୍ତ୍ରଣା, ପୋଡ଼ିଯିବା...",
        "send": "ଜରୁରୀକାଳୀନ ଆଲର୍ଟ ପଠାନ୍ତୁ",
        "detected": "🗣️ ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି:",
        "action_header": "ପ୍ରଥମ ଚିକିତ୍ସା ନିର୍ଦ୍ଦେଶାବଳୀ",
        "first_aid_label": "ତୁରନ୍ତ ପ୍ରଥମ ଚିକିତ୍ସା ପଦକ୍ଷେପ:",
        "call_btn": "ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ",
        "directory_header": "ଜରୁରୀକାଳୀନ ଡାଇରେକ୍ଟୋରୀ",
        "helpline": "ହେଲ୍ପଲାଇନ୍:",
        "audio_response": "🔊 ସହାୟ ଭଏସ୍ ମାର୍ଗଦର୍ଶନ:",
    },
    "mni_IN": {
        "title": "ꯁꯍꯥꯌ (Sahaya)",
        "subtitle": "ꯉꯥꯛꯊꯣꯛꯆꯕꯤꯌꯨ 🛡️",
        "mic_header": "🎙️ ꯋꯥ ꯉꯥꯡꯅꯕ ꯃꯥꯏꯛ ꯇꯦꯞ ꯇꯧꯕꯤꯌꯨ",
        "type_header": "⌨️ ꯈꯨꯗꯣꯡꯊꯤꯕ ꯏꯕꯤꯌꯨ",
        "type_placeholder": "ꯑꯦꯛꯁꯤꯗꯦꯟꯠ, ꯊꯃꯣꯏ ꯆꯤꯛꯄ, ꯃꯩ ꯆꯥꯛꯄ...",
        "send": "ꯈꯨꯗꯣꯡꯊꯤꯕꯒꯤ ꯄꯥꯎ ꯊꯥꯕꯤꯌꯨ",
        "detected": "🗣️ ꯈꯪꯗꯣꯛꯂꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ:",
        "action_header": "ꯑꯍꯥꯟꯕ ꯃꯇꯦꯡ (First Aid)",
        "first_aid_label": "ꯈꯨꯗꯛꯇ ꯇꯧꯒꯗꯕ ꯊꯕꯛꯁꯤꯡ:",
        "call_btn": "{number} ꯗ ꯍꯧꯖꯤꯛꯃꯛ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ",
        "directory_header": "ꯑꯇꯣꯞꯄ ꯃꯇꯦꯡ ꯅꯝꯕꯔꯁꯤꯡ",
        "helpline": "ꯍꯦꯜꯞꯂꯥꯏꯟ:",
        "audio_response": "🔊 ꯁꯍꯥꯌꯒꯤ ꯄꯥꯎꯈꯨꯝ:",
    },
}

# ---------------------------------------------------------------------------
# Emergency Directory
# ---------------------------------------------------------------------------
EMERGENCY_DIRECTORY = [
    {
        "title": "Medical & Ambulance",
        "icon": "🚑",
        "number": "108",
        "desc": "Medical emergencies, accidents, cardiac arrest, trauma & urgent patient dispatch.",
    },
    {
        "title": "Police & Safety",
        "icon": "👮",
        "number": "112",
        "desc": "Immediate law enforcement, road crash clearance & accident site protection.",
    },
    {
        "title": "Women Distress Helpline",
        "icon": "🛡️",
        "number": "1091",
        "desc": "Direct protection, helpline assistance and intervention against harassment.",
    },
    {
        "title": "Fire & Rescue",
        "icon": "🚒",
        "number": "101",
        "desc": "Fire hazard suppression, severe burns mitigation and trapped vehicle extraction.",
    },
    {
        "title": "Disaster Support",
        "icon": "🌊",
        "number": "1078",
        "desc": "Relief and evacuation assistance during floods, earthquakes and storms.",
    },
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}

# ---------------------------------------------------------------------------
# Script Ranges & Romanized Keywords for All 12 Languages
# ---------------------------------------------------------------------------
SCRIPT_RANGES = [
    (0x0C80, 0x0CFF, "kn_IN"),  # Kannada
    (0x0900, 0x097F, "hi_IN"),  # Devanagari (Hindi/Marathi)
    (0x0B80, 0x0BFF, "ta_IN"),  # Tamil
    (0x0C00, 0x0C7F, "te_IN"),  # Telugu
    (0x0D00, 0x0D7F, "ml_IN"),  # Malayalam
    (0x0980, 0x09FF, "bn_IN"),  # Bengali
    (0x0A80, 0x0AFF, "gu_IN"),  # Gujarati
    (0x0A00, 0x0A7F, "pa_IN"),  # Punjabi
    (0x0B00, 0x0B7F, "or_IN"),  # Odia
    (0xABC0, 0xABFF, "mni_IN"), # Meitei Mayek
]

ROMAN_LANG_KEYWORDS = {
    "kn_IN": {
        "nana", "nanna", "nanage", "namage", "tande", "taayi", "thande", "thaayi",
        "appa", "amma", "edeno", "edenoovu", "ede", "novu", "usiru", "husiradal",
        "husiru", "usiradalu", "tondare", "tandre", "aagide", "aagite", "shuruagite",
        "shuru", "agat", "aagatte", "bahla", "bahala", "haavu", "havu", "kacchide",
        "kachide", "beku", "sahaya", "madadi", "dayavittu", "rakta", "benki",
        "aspatre", "aaspatre", "othada", "kodi", "gundige", "sikakide", "biddare",
        "apaghata", "gaaya", "muridide", "churukide", "sudide"
    },
    "hi_IN": {
        "mujhe", "humein", "humko", "mere", "meri", "pita", "pitaji", "papa", "mataji",
        "seene", "dard", "saans", "takleef", "ruk", "chhati", "saanp", "saap",
        "bachao", "madad", "khoon", "chot", "ghayal", "aspatal", "jaldi", "kripya", "bhai",
        "hadsa", "jal", "gira", "toot", "haddi"
    },
    "ta_IN": {
        "enakku", "engalukku", "nenju", "vali", "swasam", "moochu", "thinaral",
        "appa", "thanthai", "amma", "pambu", "paambu", "kadichathu", "udavi",
        "vendum", "ratham", "thee", "kaapathunga", "maruthuvamanai", "seekiram",
        "vibathu", "kaayam", "murivu"
    },
    "te_IN": {
        "naaku", "maaku", "gunde", "noppi", "swasa", "kastam", "adatam", "nanna",
        "thandri", "amma", "paamu", "karichindi", "sahayam", "kavali", "raktam",
        "mantalu", "kapadandi", "aaspattri", "thondaraga", "pramadam", "gayamu"
    },
    "ml_IN": {
        "enikku", "njangalkku", "nenju", "vedana", "shwasam", "muttal", "achan",
        "amma", "paampu", "kadichu", "sahayam", "venam", "raktham", "thee",
        "rakshikku", "aashupathri", "vegam", "apadakam", "murivu"
    },
    "mr_IN": {
        "mala", "aamhaala", "majhya", "chhati", "dukhate", "dukhne", "shwas",
        "trass", "vadil", "baba", "saap", "chavla", "madat", "havi", "rakta",
        "vachva", "davakhana", "lavkar", "apghat", "jhaala"
    },
    "bn_IN": {
        "amar", "amader", "buker", "byatha", "batha", "shwas", "kosto", "baba",
        "shap", "kamreche", "sahajjo", "dorkar", "rokto", "aagun", "bachan",
        "haspatal", "taratari", "durghotona"
    },
    "gu_IN": {
        "mane", "amne", "maru", "chhati", "dard", "shwas", "takleef", "bapa",
        "saap", "dankhyo", "madad", "joiye", "lohi", "aag", "bachavo", "davakhana", "jaldi", "aksmata"
    },
    "pa_IN": {
        "meri", "mere", "sanu", "chhati", "peerh", "saah", "aukh", "bapu",
        "sapp", "dangea", "madad", "chahidi", "khoon", "agg", "bachao", "aspataal", "cheti", "haadsa"
    },
    "or_IN": {
        "mora", "aamara", "chhati", "bitha", "shwasa", "kasta", "bapa", "sapa",
        "kamudichi", "sahajya", "darkar", "rakta", "nia", "bachantu", "sighra", "durghatana"
    },
    "mni_IN": {
        "eigi", "eikhoi", "thamoigi", "chikpa", "masha", "ithak", "lin",
        "chikre", "mateng", "pibiyu", "ee", "mei", "kanbiyu", "thuna"
    },
    "en_IN": {
        "accident", "crash", "bleeding", "blood", "fracture", "bone", "burn", "burns",
        "fall", "head", "injury", "choking", "drowning", "electric", "shock", "chest",
        "heart", "pain", "breathe", "ambulance", "hospital"
    }
}

# ---------------------------------------------------------------------------
# Comprehensive Minor & Major Accidents Triage
# ---------------------------------------------------------------------------
def identify_condition(text: str) -> str:
    t = text.lower()
    
    # 1. Road Accident / Severe Trauma / Bleeding / Fractures
    if any(k in t for k in [
        "accident", "crash", "bike", "car", "hit", "run over", "bleeding", "blood",
        "fracture", "bone broke", "bone broken", "cut", "wound", "injury", "head injury",
        "apaghata", "gaaya", "rakta", "muridide", "rakthasrava",
        "haadsa", "khoon", "chot", "toot", "haddi", "ghayal",
        "vibathu", "ratham", "kaayam", "murivu",
        "pramadam", "raktam", "gayamu", "virigindi"
    ]):
        return "accident_trauma"

    # 2. Cardiac & Respiratory Emergencies
    if any(k in t for k in [
        "heart", "chest", "breath", "breathing", "cardiac", "stroke", "attack",
        "ede", "edeno", "usir", "husir", "gundige", "seene", "chhati", "saans",
        "nenju", "moochu", "swasam", "gunde", "swasa", "buker", "shwas", "peerh", "saah"
    ]):
        return "cardiac"

    # 3. Snakebite & Envenomation
    if any(k in t for k in [
        "snake", "bite", "haavu", "havu", "kachide", "saanp", "saap",
        "paambu", "paamu", "lin", "chikpa", "sapp", "sapa", "dankh"
    ]):
        return "snakebite"

    # 4. Burns & Scalds (Minor to Major)
    if any(k in t for k in [
        "fire", "burn", "burns", "hot water", "scald", "cylinder", "blast",
        "benki", "sudide", "aag", "jal gaya", "thee", "agg", "nia", "mei"
    ]):
        return "burns"

    # 5. Electric Shock
    if any(k in t for k in [
        "electric", "shock", "current", "wire", "vidyut", "karent"
    ]):
        return "electric_shock"

    # 6. Choking / Asphyxiation
    if any(k in t for k in [
        "chok", "stuck in throat", "gantlu", "adagide", "gale mein", "thொண்டை"
    ]):
        return "choking"

    # 7. Poison / Pesticide Ingestion
    if any(k in t for k in [
        "poison", "pesticide", "chemical", "visha", "keetnashaka", "zeher", "marunthu"
    ]):
        return "pesticide"

    # 8. Maternal / Delivery
    if any(k in t for k in [
        "pregnant", "pregnancy", "delivery", "labor", "maternal", "garbhini", "prasava"
    ]):
        return "maternal"

    # 9. Violence / Crime / Police
    if any(k in t for k in [
        "police", "thief", "robbery", "attack", "fight", "assault", "harassment", "chor", "bachao"
    ]):
        return "police"

    return "general_medical"

CONDITION_FALLBACKS = {
    "accident_trauma": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಅಪಘಾತ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ದಯವಿಟ್ಟು ರೋಗಿಯನ್ನು ಅತಿಯಾಗಿ ಚಲಿಸಬೇಡಿ ಮತ್ತು ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ತೀವ್ರ ರಕ್ತಸ್ರಾವವಿದ್ದರೆ, ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರವಾಗಿ ಬಲವಾದ ಒತ್ತಡ ಹಾಕಿ ಹಿಡಿಯಿರಿ.",
                "ಕುತ್ತಿಗೆ ಅಥವಾ ಬೆನ್ನುಮೂಳೆಗೆ ಪೆಟ್ಟಾಗಿದ್ದರೆ, ರೋಗಿಯನ್ನು ಎತ್ತಬೇಡಿ ಅಥವಾ ತಿರುಗಿಸಬೇಡಿ (ಬೆನ್ನಿನ ಸ್ಥಿರತೆ ಮುಖ್ಯ).",
                "ಮೂಳೆ ಮುರಿದಿದ್ದರೆ (Fracture), ಆ ಜಾಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಸ್ಪ್ಲಿಂಟ್ ಅಥವಾ ಮರದ ತುಂಡಿನಿಂದ ಸ್ಥಿರವಾಗಿರಿಸಿ.",
                "ಗಾಯಾಳುವಿಗೆ ಕುಡಿಯಲು ನೀರು ಅಥವಾ ಆಹಾರ ನೀಡಬೇಡಿ. ತಕ್ಷಣ 108 ಬರುವವರೆಗೆ ಜೊತೆಯಲ್ಲೇ ಇರಿ."
            ]
        ),
        "hi_IN": (
            "दुर्घटना / चोट की आपात स्थिति! मरीज को ज्यादा न हिलाएं और तुरंत 108 एम्बुलेंस को कॉल करें।",
            [
                "यदि खून बह रहा है, तो साफ कपड़े से घाव पर सीधा और मजबूत दबाव बनाए रखें।",
                "यदि रीढ़ या गर्दन में चोट की आशंका है, तो मरीज को बिल्कुल न हिलाएं।",
                "टूटी हुई हड्डी को हिलाने की कोशिश न करें, उसे सीधा सहारा देकर रखें।",
                "घायल व्यक्ति को कुछ भी खाने या पीने को न दें और 108 का इंतजार करें।"
            ]
        ),
        "en_IN": (
            "Accident & Trauma Emergency! Keep the patient still and dial 108 immediately.",
            [
                "For severe bleeding: Apply firm, direct pressure over the wound with a clean cloth or gauze.",
                "Do NOT move the patient if head, neck, or spine injuries are suspected.",
                "If a bone is fractured: Keep the injured limb completely still and supported.",
                "Do not give food or water to the injured person. Await 108 emergency paramedics."
            ]
        ),
        "ta_IN": (
            "விபத்து அவசரநிலை! நோயாளியை அசைக்காமல் உடனே 108 ஆம்புலன்ஸை அழைக்கவும்.",
            [
                "அதிக ரத்தப்போக்கு இருந்தால், சுத்தமான துணியால் காயத்தின் மீது அழுத்திப் பிடிக்கவும்.",
                "கழுத்து அல்லது முதுகெலும்பில் அடிபட்டிருந்தால் நோயாளியை நகர்த்த வேண்டாம்.",
                "எலும்பு முறிவு ஏற்பட்டால் அந்தப் பகுதியை அசைக்காமல் நிலைநிறுத்தவும்.",
                "உடனடியாக 108 மருத்துவக் குழு வரும் வரை உடனிருங்கள்."
            ]
        ),
        "te_IN": (
            "ప్రమాద అత్యవసర పరిస్థితి! రోగిని కదల్చకండి, వెంటనే 108 అంబులెన్స్‌కు కాల్ చేయండి.",
            [
                "తీవ్ర రక్తస్రావం ఉంటే శుభ్రమైన గుడ్డతో గాయంపై గట్టిగా ఒత్తిడి ఉంచండి.",
                "మెడ లేదా వెన్నుముక గాయాలుంటే రోగిని అస్సలు కదల్చవద్దు.",
                "ఎముక విరిగిన భాగాన్ని కదలకుండా స్థిరంగా ఉంచండి.",
                "గాయపడిన వారికి ఏమీ తినిపించవద్దు లేదా తాగించవద్దు. 108 కోసం వేచి ఉండండి."
            ]
        ),
        "ml_IN": (
            "അപകട അടിയന്തരാവസ്ഥ! രോഗിയെ അനക്കാതെ ഉടൻ 108 ആംബുലൻസ് വിളിക്കുക.",
            [
                "രക്തസ്രാവമുണ്ടെങ്കിൽ വൃത്തിയുള്ള തുണികൊണ്ട് മുറിവിൽ അമർത്തിപ്പിടിക്കുക.",
                "കഴുത്തിനോ നട്ടെല്ലിനോ പരിക്കേറ്റിട്ടുണ്ടെങ്കിൽ രോഗിയെ മാറ്റരുത്.",
                "അസ്ഥി ഒടിഞ്ഞ ഭാഗം അനങ്ങാതെ താങ്ങിനിർത്തുക.",
                "ഉടൻ തന്നെ 108 എത്തിച്ചേരുന്നതുവരെ ഒപ്പം തുടരുക."
            ]
        ),
        "mr_IN": (
            "अपघात आणीबाणी! रुग्णाला जास्त हलवू नका आणि त्वरित 108 रुग्णवाहिकेला कॉल करा.",
            [
                "रक्तस्त्राव होत असल्यास स्वच्छ कापडाने जखमेवर थेट दाब द्या.",
                "मान किंवा पाठीच्या कण्याला इजा झाली असल्यास रुग्णाला हलवू नका.",
                "हाड मोडले असल्यास ते स्थिर ठेवा.",
                "रुग्णाला पाणी किंवा अन्न देऊ नका, त्वरित 108 ची वाट पहा."
            ]
        ),
        "gu_IN": (
            "અકસ્માત કટોકટી! દર્દીને બિલકુલ હલાવશો નહીં અને તરત જ 108 પર કૉલ કરો.",
            [
                "જો લોહી વહેતું હોય, તો સાફ કપડાથી ઘા પર સીધું દબાણ આપો.",
                "ગરદન કે કરોડરજ્જુની ઇજા હોય તો દર્દીને જરાય ખસેડશો નહીં.",
                "હાડકું તૂટી ગયું હોય તો તે ભાગને સ્થિર રાખો.",
                "108 એમ્બ્યુલન્સ આવે ત્યાં સુધી દર્દી સાથે રહો."
            ]
        ),
        "bn_IN": (
            "দুর্ঘটনাজনিত জরুরি অবস্থা! রোগীকে বেশি নাড়াচাড়া করবেন না এবং অবিলম্বে 108-এ কল করুন।",
            [
                "মারাত্মক রক্তপাত হলে পরিষ্কার কাপড় দিয়ে ক্ষতের ওপর শক্ত করে চেপে ধরুন।",
                "ঘাড় বা মেরুদণ্ডে আঘাতের সম্ভাবনা থাকলে রোগীকে সরাবেন না।",
                "হাড় ভেঙে গেলে সেই অংশটি সম্পূর্ণ স্থির রাখুন।",
                "আহত ব্যক্তিকে খাবার বা জল দেবেন না। 108-এর জন্য অপেক্ষা করুন।"
            ]
        ),
        "pa_IN": (
            "ਹਾਦਸਾ ਐਮਰਜੈਂਸੀ! ਮਰੀਜ਼ ਨੂੰ ਨਾ ਹਿਲਾਓ ਅਤੇ ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਨੂੰ ਕਾਲ ਕਰੋ।",
            [
                "ਜੇ ਖੂਨ ਵਹਿ ਰਿਹਾ ਹੈ, ਤਾਂ ਸਾਫ਼ ਕੱਪੜੇ ਨਾਲ ਜ਼ਖ਼ਮ 'ਤੇ ਦਬਾਅ ਬਣਾਓ।",
                "ਧੌਣ ਜਾਂ ਰੀੜ੍ਹ ਦੀ ਹੱਡੀ ਦੀ ਸੱਟ ਲੱਗਣ 'ਤੇ ਮਰੀਜ਼ ਨੂੰ ਬਿਲਕੁਲ ਨਾ ਹਿਲਾਓ।",
                "ਟੁੱਟੀ ਹੋਈ ਹੱਡੀ ਨੂੰ ਸਹਾਰਾ ਦੇ ਕੇ ਸਥਿਰ ਰੱਖੋ।",
                "ਤੁਰੰਤ 108 ਐਂਬੂਲੈਂਸ ਦੀ ਉਡੀਕ ਕਰੋ।"
            ]
        ),
        "or_IN": (
            "ଦୁର୍ଘଟଣା ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି! ରୋଗୀଙ୍କୁ ନ ହଲାଇ ତୁରନ୍ତ 108 ଆମ୍ବୁଲାନ୍ସକୁ କଲ୍ କରନ୍ତୁ।",
            [
                "ରକ୍ତସ୍ରାବ ହେଉଥିଲେ ପରିଷ୍କାର କପଡ଼ାରେ କ୍ଷତ ସ୍ଥାନକୁ ଜୋରରେ ଚାପି ଧରନ୍ତୁ।",
                "ବେକ କିମ୍ବା ମେରୁଦଣ୍ଡରେ ଆଘାତ ଥିଲେ ରୋଗୀଙ୍କୁ ଆଦୌ ଘୁଞ୍ଚାନ୍ତୁ ନାହିଁ।",
                "ଭଙ୍ଗା ହାଡ଼କୁ ସ୍ଥିର ରଖନ୍ତୁ।",
                "ତୁରନ୍ତ 108 ର ସହାୟତା ନିଅନ୍ତୁ।"
            ]
        ),
        "mni_IN": (
            "ꯑꯦꯛꯁꯤꯗꯦꯟꯠ ꯑꯆꯧꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ! ꯑꯅꯥꯕ ꯃꯤꯑꯣꯏ ꯑꯗꯨ ꯍꯥꯏꯗꯣꯛ-ꯍꯥꯏꯖꯤꯟ ꯇꯧꯒꯅꯨ, 108 ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ।",
            [
                "ꯏ ꯌꯥꯝꯅ ꯊꯣꯛꯂꯕꯗꯤ ꯂꯨꯅ ꯂꯩꯕ ꯐꯤꯅ ꯀꯟꯅ ꯅꯝꯁꯤꯟꯕꯤꯌꯨ।",
                "ꯀꯣꯛ ꯑꯃꯁꯨꯡ ꯉꯛ ꯁꯣꯛꯂꯕꯗꯤ ꯃꯤꯑꯣꯏ ꯑꯗꯨꯕꯨ ꯍꯤꯡꯁꯥꯠ ꯇꯧꯍꯟꯒꯅꯨ।",
                "ꯁꯔꯨ ꯊꯨꯒꯥꯏꯔꯕꯗꯤ ꯆꯠꯊꯣꯛ-ꯆꯠꯁꯤꯟ ꯇꯧꯍꯟꯒꯅꯨ।",
                "108 ꯂꯥꯛꯇ꯭ꯔꯤꯐꯥꯎ ꯃꯇꯦꯡ ꯄꯥꯡꯕꯤꯌꯨ।"
            ]
        ),
    },
    "burns": {
        "service_idx": 3,
        "urgency": "critical",
        "kn_IN": (
            "ಸುಟ್ಟ ಗಾಯದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 101/108 ಕರೆ ಮಾಡಿ ಮತ್ತು ಈ ಕ್ರಮಗಳನ್ನು ಮಾಡಿ.",
            [
                "ಸುಟ್ಟ ಜಾಗದ ಮೇಲೆ ಕನಿಷ್ಠ 15-20 ನಿಮಿಷಗಳ ಕಾಲ ನಿರಂತರವಾಗಿ ತಣ್ಣನೆಯ ಹರಿಯುವ ನೀರನ್ನು ಹಾಕಿ (ಮಂಜುಗಡ್ಡೆ ಬಳಸಬೇಡಿ).",
                "ಸುಟ್ಟ ಚರ್ಮಕ್ಕೆ ಅಂಟಿಕೊಂಡಿರುವ ಬಟ್ಟೆಗಳನ್ನು ಬಲವಂತವಾಗಿ ಎಳೆಯಬೇಡಿ.",
                "ಗಾಯದ ಮೇಲೆ ಟೂತ್‌ಪೇಸ್ಟ್, ಎಣ್ಣೆ ಅಥವಾ ಬೆಣ್ಣೆಯನ್ನು ಎಂದಿಗೂ ಹಚ್ಚಬೇಡಿ.",
                "ಸ್ವಚ್ಛವಾದ ಒಣ ಬಟ್ಟೆಯಿಂದ ಮುಚ್ಚಿ ತಕ್ಷಣ ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಿರಿ."
            ]
        ),
        "hi_IN": (
            "जलने की आपात स्थिति! तुरंत ठंडे पानी का उपयोग करें और 108/101 पर कॉल करें।",
            [
                "जले हुए स्थान पर कम से कम 15-20 मिनट तक सामान्य ठंडा बहता पानी डालें (बर्फ न लगाएं)।",
                "जली हुई त्वचा से चिपके कपड़ों को जबरदस्ती न खींचें।",
                "टूथपेस्ट, तेल या हल्दी जैसी चीजें बिल्कुल न लगाएं।",
                "साफ सूखे कपड़े से ढककर तुरंत अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Burns & Fire Emergency! Cool the burn immediately and call 108/101.",
            [
                "Cool the burn under clean, cool running water for at least 15-20 minutes. Do NOT use ice.",
                "Do NOT peel away clothing stuck to burned skin.",
                "Never apply toothpaste, butter, oils, or home remedies on burn wounds.",
                "Cover loosely with a clean, sterile cloth and transport to the burns emergency unit."
            ]
        ),
    },
    "cardiac": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಎದೆನೋವು / ಹೃದಯ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ ಮತ್ತು ಈ ಕ್ರಮಗಳನ್ನು ಪಾಲಿಸಿ.",
            [
                "ವ್ಯಕ್ತಿಯನ್ನು ಆರಾಮವಾಗಿ ಕುಳ್ಳಿರಿಸಿ ಮತ್ತು ಕುತ್ತಿಗೆ ಹಾಗೂ ಎದೆಯ ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ.",
                "ರೋಗಿಯನ್ನು ಸಂಪೂರ್ಣ ಶಾಂತವಾಗಿರಲು ಹೇಳಿ, ಯಾವುದೇ ಶ್ರಮ ಅಥವಾ ನಡಿಗೆಗೆ ಬಿಡಬೇಡಿ.",
                "ಪ್ರಜ್ಞೆ ತಪ್ಪಿ ಉಸಿರಾಟ ನಿಂತಿದ್ದರೆ, ತಕ್ಷಣ ಎದೆಯ ಮಧ್ಯಭಾಗದಲ್ಲಿ ಸಿಪಿಆರ್ (CPR) ಒತ್ತಡ ಪ್ರಾರಂಭಿಸಿ (ನಿಮಿಷಕ್ಕೆ 100-120 ಬಾರಿ).",
                "ತಕ್ಷಣವೇ ಸಮೀಪದ ತುರ್ತು ಹೃದಯ ಚಿಕಿತ್ಸಾ ಆಸ್ಪತ್ರೆಗೆ ರವಾನಿಸಿ."
            ]
        ),
        "hi_IN": (
            "सीने में दर्द / हार्ट अटैक की आपात स्थिति! तुरंत 108 पर कॉल करें।",
            [
                "मरीज को आराम से बैठाएं और गले व कमर के कपड़े ढीले करें।",
                "मरीज को शांत रखें और कोई भी शारीरिक श्रम न करने दें।",
                "यदि सांस रुक जाए, तो तुरंत सीपीआर (CPR) शुरू करें (100-120 प्रति मिनट)।",
                "तुरंत 108 पर संपर्क कर नजदीकी अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Cardiac & Chest Pain Emergency! Dial 108 immediately and follow these steps.",
            [
                "Sit the person down comfortably and loosen tight clothing around the neck and waist.",
                "Keep the patient calm and still. Do not let them walk or exert themselves.",
                "If the person becomes unresponsive and stops breathing normally, begin CPR immediately (100-120 compressions/min).",
                "Transport immediately to the nearest hospital with emergency cardiac facilities."
            ]
        ),
    },
    "snakebite": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ಹಾವು ಕಚ್ಚಿದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ, ತಕ್ಷಣ 108 ಕರೆ ಮಾಡಿ.",
            [
                "ಕಚ್ಚಿದ ಅಂಗವನ್ನು ಅಲುಗಾಡಿಸದೆ ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗೆ ಇರಿಸಿ.",
                "ಗಟ್ಟಿಯಾಗಿ ಹಗ್ಗ ಕಟ್ಟಬೇಡಿ, ಗಾಯವನ್ನು ಕತ್ತರಿಸಬೇಡಿ ಅಥವಾ ವಿಷ ಹೀರುವ ಪ್ರಯತ್ನ ಮಾಡಬೇಡಿ.",
                "ಆಂಟಿ-ವೆನಮ್ (ASV) ಲಭ್ಯವಿರುವ ಸಮೀಪದ ಆಸ್ಪತ್ರೆಗೆ ತಕ್ಷಣ ಕರೆದೊಯ್ಯಿರಿ."
            ]
        ),
        "hi_IN": (
            "सांप के काटने पर तुरंत 108 पर कॉल करें और शांत रहें।",
            [
                "काटे गए अंग को हिलाए बिना दिल के स्तर से नीचे रखें।",
                "कसकर पट्टी न बांधें, चीरा न लगाएं और जहर चूसने की कोशिश न करें।",
                "मरीज को तुरंत एंटी-स्नेक वेनम वाले अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Snakebite emergency! Keep the victim still and calm, dial 108 immediately.",
            [
                "Immobilize the bitten limb and keep it below heart level.",
                "DO NOT apply a tight tourniquet, do not cut the wound, and do not suck venom.",
                "Rush immediately to the nearest hospital equipped with Anti-Snake Venom (ASV)."
            ]
        ),
    },
    "electric_shock": {
        "service_idx": 0,
        "urgency": "critical",
        "kn_IN": (
            "ವಿದ್ಯುತ್ ಆಘಾತ (Electric Shock)! ರೋಗಿಯನ್ನು ಬರಿಗೈಯಿಂದ ಮುಟ್ಟಬೇಡಿ, ತಕ್ಷಣ ಮೇನ್ ಸ್ವಿಚ್ ಆಫ್ ಮಾಡಿ.",
            [
                "ಮೊದಲು ವಿದ್ಯುತ್ ಸಂಪರ್ಕ ಕಡಿತಗೊಳಿಸಿ ಅಥವಾ ಒಣ ಮರದ ತುಂಡಿನಿಂದ ವ್ಯಕ್ತಿಯನ್ನು ತಂತಿಯಿಂದ ಬೇರ್ಪಡಿಸಿ.",
                "ಉಸಿರಾಟ ಮತ್ತು ನಾಡಿಮಿಡಿತ ಪರೀಕ್ಷಿಸಿ. ಉಸಿರಾಟವಿಲ್ಲದಿದ್ದರೆ ಸಿಪಿಆರ್ (CPR) ಪ್ರಾರಂಭಿಸಿ.",
                "ಸುಟ್ಟ ಗಾಯವಿದ್ದರೆ ಶುದ್ಧ ಬಟ್ಟೆಯಿಂದ ಮುಚ್ಚಿ ತಕ್ಷಣ 108 ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಿರಿ."
            ]
        ),
        "hi_IN": (
            "बिजली का झटका (Electric Shock)! मरीज को सीधे न छुएं, पहले बिजली बंद करें।",
            [
                "मेन स्विच बंद करें या सूखी लकड़ी की मदद से व्यक्ति को तार से अलग करें।",
                "सांस की जांच करें। सांस न चलने पर तुरंत सीपीआर (CPR) दें।",
                "तुरंत 108 पर कॉल करें और अस्पताल ले जाएं।"
            ]
        ),
        "en_IN": (
            "Electric Shock Emergency! Do not touch the victim with bare hands.",
            [
                "Immediately turn off the main power source or use a dry wooden object to separate the victim from the current.",
                "Check breathing and responsiveness. Begin CPR if the person is not breathing.",
                "Dial 108 immediately for urgent paramedic assistance."
            ]
        ),
    },
    "general_medical": {
        "service_idx": 0,
        "urgency": "urgent",
        "kn_IN": (
            "ತುರ್ತು ಪರಿಸ್ಥಿತಿ! ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ.",
            [
                "ವ್ಯಕ್ತಿಯ ಉಸಿರಾಟ ಮತ್ತು ಪ್ರಜ್ಞೆಯನ್ನು ನಿರಂತರವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
                "ಗಾಯ ಅಥವಾ ರಕ್ತಸ್ರಾವವಿದ್ದರೆ ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ನೇರ ಒತ್ತಡ ಹಾಕಿ.",
                "ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ರೋಗಿಯೊಂದಿಗೆ ಶಾಂತವಾಗಿ ಇರಿ."
            ]
        ),
        "hi_IN": (
            "आपात स्थिति! शांत रहें और तुरंत 108 पर कॉल करें।",
            [
                "व्यक्ति की सांस और होश की जांच करें।",
                "घाव पर साफ कपड़े से सीधा दबाव बनाकर रखें।",
                "तुरंत एम्बुलेंस की प्रतीक्षा करें।"
            ]
        ),
        "en_IN": (
            "Medical Emergency! Please stay calm and immediately call 108.",
            [
                "Check airway, responsiveness, and normal breathing continuously.",
                "Apply clean, firm pressure to any actively bleeding wounds.",
                "Stay with the patient until the emergency medical team arrives."
            ]
        ),
    }
}

# ---------------------------------------------------------------------------
# Language & Response Detection
# ---------------------------------------------------------------------------
def get_env(name):
    return os.getenv(name, "").strip()

def to_hyphen(code):
    return code.replace("_", "-")

def detect_lang(text: str, fallback: str) -> str:
    if not text:
        return fallback

    # 1. Native script detection
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        best_script = max(counts, key=counts.get)
        if best_script == "hi_IN" and fallback == "mr_IN":
            return "mr_IN"
        return best_script

    # 2. Romanized transliterated detection
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    scores = {}
    for code, kw_set in ROMAN_LANG_KEYWORDS.items():
        matched = words & kw_set
        if matched:
            scores[code] = len(matched)

    if scores:
        best_code = max(scores, key=scores.get)
        if fallback in scores and scores[fallback] >= scores[best_code] - 1:
            return fallback
        return best_code

    return fallback

def get_fallback_response(transcript: str, lang_code: str):
    cond = identify_condition(transcript)
    cond_data = CONDITION_FALLBACKS.get(cond, CONDITION_FALLBACKS["accident_trauma"])
    service = EMERGENCY_DIRECTORY[cond_data["service_idx"]]
    urgency = cond_data["urgency"]

    if lang_code in cond_data:
        reply, steps = cond_data[lang_code]
    elif "en_IN" in cond_data:
        reply, steps = cond_data["en_IN"]
    else:
        reply, steps = CONDITION_FALLBACKS["accident_trauma"]["en_IN"]

    return service, reply, steps, urgency

# ---------------------------------------------------------------------------
# AI Engine: STT, Groq Reasoning, TTS
# ---------------------------------------------------------------------------
def transcribe(audio_path, lang_code):
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return "", "GNANI_API_KEY is missing."

    stt_lang = "en-IN" if lang_code == "mni_IN" else to_hyphen(lang_code)
    try:
        with open(audio_path, "rb") as f:
            r = requests.post(
                STT_URL,
                headers={"X-API-Key-ID": api_key},
                files={"audio_file": ("voice.wav", f, "audio/wav")},
                data={"language_code": stt_lang, "format": "transcribe"},
                timeout=30,
            )
    except Exception as e:
        return "", f"Speech connection failed: {e}"

    if r.status_code != 200:
        return "", f"Speech recognition failed ({r.status_code})"

    try:
        data = r.json()
    except ValueError:
        return "", "Invalid response from speech server."

    transcript = str(data.get("transcript") or data.get("text") or "").strip()
    return transcript, ""

def groq_triage(transcript: str, detected_lang: str):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "Groq key missing"

    system_prompt = f"""You are Sahaya, an empathetic AI emergency first-aid dispatcher in India.
Your mission is to handle all minor and major emergencies (road accidents, severe bleeding, fractures, burns, cardiac arrest, snakebites, electric shock, choking, etc.).

CRITICAL RULES:
1. DETECT INTENDED LANGUAGE:
   - Identify whether the caller used Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Manipuri, or English.
   - If Romanized Kannada is detected (e.g. 'nana tande...', 'apaghata...'), set "lang_code": "kn_IN".

2. RESPONSE MUST BE IN THE EXACT NATIVE SCRIPT:
   - Respond strictly in the native script of that detected language.
   - NEVER reply in Hindi or English if Kannada was spoken!

3. ACCURATE FIRST-AID:
   - For road accidents: direct pressure on bleeding, do not move neck/spine, immobilize fractures, call 108.
   - For burns: 20 mins clean cool running water, no ice/toothpaste.
   - For cardiac: loosen clothing, CPR if unresponsive, call 108.

4. RETURN ONLY STRICT RAW JSON:
{{
  "lang_code": "{detected_lang}",
  "category": "medical",
  "urgency": "critical",
  "reply": "Calm supportive response with call-to-action in native script",
  "steps": [
    "Step 1 in native script",
    "Step 2 in native script",
    "Step 3 in native script",
    "Step 4 in native script"
  ]
}}"""

    custom_model = get_env("GROQ_MODEL")
    models_to_try = ([custom_model] if custom_model else []) + [
        m for m in GROQ_MODELS if m != custom_model
    ]

    for model_name in models_to_try:
        try:
            r = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {groq_key}",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Emergency: {transcript}"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 550,
                },
                timeout=20,
            )
            if r.status_code != 200:
                continue

            content = r.json()["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            clean_json = re.sub(r"```(json)?", "", content).strip()
            start, end = clean_json.find("{"), clean_json.rfind("}")
            if start == -1 or end <= start:
                continue

            data = json.loads(clean_json[start : end + 1])
            reply = str(data.get("reply", "")).strip()
            if not reply:
                continue

            steps = [
                re.sub(r"^\d+\.\s*", "", str(s).strip())
                for s in data.get("steps", [])
                if str(s).strip()
            ][:5]

            det_lang = str(data.get("lang_code", detected_lang)).strip()
            if det_lang not in TRANSLATIONS:
                det_lang = detected_lang

            return {
                "lang_code": det_lang,
                "category": str(data.get("category", "medical")).strip().lower(),
                "urgency": str(data.get("urgency", "critical")).strip().lower(),
                "reply": reply,
                "steps": steps,
            }, ""
        except Exception:
            continue

    return None, "Offline fallback active"

def timbre_speak(text: str, lang_code: str):
    if lang_code not in TIMBRE_LANGS:
        return None, "Voice not supported for this dialect"
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return None, "API key missing"

    try:
        r = requests.post(
            TTS_URL,
            headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "voice": get_env("TIMBRE_VOICE") or "Shivani",
                "model": get_env("TIMBRE_MODEL") or "timbre-v3.0",
                "language": to_hyphen(lang_code),
                "speed": 1.0,
                "audio_config": {
                    "sample_rate": 24000,
                    "num_channels": 1,
                    "sample_width": 2,
                    "encoding": "linear_pcm",
                    "container": "wav",
                },
            },
            timeout=25,
        )
        if r.status_code != 200:
            return None, "TTS failed"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(r.content)
            return tmp.name, ""
    except Exception:
        return None, "TTS error"

# ---------------------------------------------------------------------------
# High-Contrast UI HTML Builders
# ---------------------------------------------------------------------------
def header_html(t):
    return (
        '<div class="brand-top">'
        '  <div class="brand-title-box">'
        f'   <h1>{html.escape(t["title"])}</h1>'
        '  </div>'
        f' <p class="brand-sub">{html.escape(t["subtitle"])}</p>'
        '</div>'
    )

def result_card(service, steps, urgency, rt):
    body = "<ol class='step-list'>" + "".join(f"<li>{html.escape(s)}</li>" for s in steps) + "</ol>"
    urgency_tag = f'<span class="badge badge-{urgency}">{urgency.upper()}</span>' if urgency else ""
    return (
        '<div class="emergency-banner card-box">'
        f' <div class="emergency-icon-wrap">{service["icon"]}</div>'
        '  <div class="emergency-info">'
        f'   <div class="emergency-title-row"><h2>{html.escape(service["title"])}</h2>{urgency_tag}</div>'
        f'   <p class="service-desc">{html.escape(service["desc"])}</p>'
        '  </div>'
        '</div>'
        '<div class="content-card card-box">'
        f' <div class="card-heading">💊 {html.escape(rt["first_aid_label"])}</div>'
        f' {body}'
        f' <a class="call-action-btn" href="tel:{service["number"]}">'
        f'   📞 {html.escape(rt["call_btn"].format(number=service["number"]))}'
        '  </a>'
        '</div>'
    )

def directory_html(t):
    cards = []
    for srv in EMERGENCY_DIRECTORY:
        cards.append(
            '<div class="dir-item card-box">'
            f' <div class="dir-icon">{srv["icon"]}</div>'
            '  <div class="dir-meta">'
            f'   <h4>{html.escape(srv["title"])}</h4>'
            f'   <span class="dir-number">{srv["number"]}</span>'
            '  </div>'
            f' <a class="quick-call-btn" href="tel:{srv["number"]}">Call</a>'
            '</div>'
        )
    return (
        f'<div class="dir-container"><h3 class="dir-title">📋 {html.escape(t["directory_header"])}</h3>'
        f'<div class="dir-list">{"".join(cards)}</div></div>'
    )

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
def run_pipeline(transcript: str, selected_lang: str):
    reply_lang = detect_lang(transcript, selected_lang)
    rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["en_IN"])
    transcript_bubble = f"**{rt['detected']}**\n\n💬 {transcript}"
    yield "", transcript_bubble, "", None, ""

    triage, _ = groq_triage(transcript, reply_lang)

    if triage:
        reply_lang = triage.get("lang_code", reply_lang)
        rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["en_IN"])
        transcript_bubble = f"**{rt['detected']}**\n\n💬 {transcript}"
        service = EMERGENCY_DIRECTORY[CATEGORY_INDEX.get(triage["category"], 0)]
        reply_text = triage["reply"]
        steps = triage["steps"]
        urgency = triage["urgency"]
    else:
        service, reply_text, steps, urgency = get_fallback_response(transcript, reply_lang)

    reply_bubble = f"**{rt['audio_response']}**\n\n{reply_text}"
    card = result_card(service, steps, urgency, rt)
    yield "", transcript_bubble, reply_bubble, None, card

    # Spoken narration in the exact language
    steps_narration = " ".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    spoken_text = f"{reply_text}. {steps_narration}"
    audio_path, _ = timbre_speak(spoken_text, reply_lang)
    yield "", transcript_bubble, reply_bubble, audio_path, card

def on_audio(audio_path, lang_code):
    if not audio_path:
        yield "⚠️ Please record your voice message first.", "", "", None, ""
        return
    yield "", "", "", None, ""
    transcript, err = transcribe(audio_path, lang_code)
    if err:
        yield f"⚠️ {err}", "", "", None, ""
        return
    yield from run_pipeline(transcript, lang_code)

def on_text(text, lang_code):
    text = (text or "").strip()
    if not text:
        yield "⚠️ Please enter your message first.", "", "", None, ""
        return
    yield from run_pipeline(text, lang_code)

def apply_language(lang_code):
    t = TRANSLATIONS.get(lang_code, TRANSLATIONS["en_IN"])
    return (
        header_html(t),
        gr.update(label=t["mic_header"]),
        gr.update(label=t["type_header"], placeholder=t["type_placeholder"]),
        gr.update(value=t["send"]),
        directory_html(t),
    )

# ---------------------------------------------------------------------------
# High-Contrast Clean CSS
# ---------------------------------------------------------------------------
CSS = """
:root {
  --body-background-fill: #FFFDF9 !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F8F5EE !important;
  --block-background-fill: #FFFFFF !important;
  --block-border-color: #E2DCD2 !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: #C8BFB0 !important;
  --body-text-color: #0F172A !important;
  --block-label-text-color: #0F172A !important;
}

body, .gradio-container {
  background-color: #FFFDF9 !important;
  color: #0F172A !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  max-width: 900px !important;
  margin: 0 auto !important;
  padding: 1.2rem 1rem !important;
}
footer { display: none !important; }

.brand-top {
  text-align: center;
  padding: 1.8rem 1.4rem;
  border-radius: 18px !important;
  margin-bottom: 1rem;
  border: 1.5px solid #BAE6FD !important;
  background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%) !important;
}
.brand-top h1 { font-size: 2.4rem; font-weight: 800; margin: 0; color: #0284C7 !important; }
.brand-sub { font-size: 1.1rem; font-weight: 700; color: #334155 !important; margin-top: 4px; }

.sos-row { text-align: center; margin: 0.4rem 0 1.2rem; }
.sos-panic-btn {
  background: #DC2626 !important;
  color: #FFFFFF !important;
  font-size: 1.05rem;
  font-weight: 800;
  padding: 14px 32px;
  border-radius: 50px;
  border: none !important;
  box-shadow: 0 6px 18px rgba(220, 38, 38, 0.3);
  cursor: pointer;
}

/* Full Width Audio & Input Blocks */
.gradio-container .block, .card-box {
  background: #FFFFFF !important;
  border: 1.5px solid #E2DCD2 !important;
  border-radius: 14px !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
}

/* Audio Widget Fix: Clean, High Visibility */
.audio-full-container {
  margin-bottom: 1rem !important;
}
.audio-full-container .gr-audio {
  background: #FFFFFF !important;
  border: 1.5px solid #CBD5E1 !important;
  border-radius: 12px !important;
  padding: 10px !important;
}

/* HIGH CONTRAST STEPS FIX: Makes list text bold, crisp, and 100% visible */
.content-card {
  padding: 1.6rem 2rem !important;
  margin-top: 1rem !important;
  background: #FFFFFF !important;
}
.card-heading {
  font-size: 1.3rem !important;
  font-weight: 800 !important;
  color: #0F172A !important;
  margin-bottom: 1rem !important;
}
.step-list {
  margin: 0.8rem 0 1.4rem !important;
  padding-left: 1.8rem !important;
}
.step-list li {
  color: #0F172A !important;
  font-size: 1.12rem !important;
  font-weight: 700 !important;
  line-height: 1.8 !important;
  margin-bottom: 0.6rem !important;
  opacity: 1 !important;
}

.emergency-banner {
  display: flex;
  align-items: center;
  gap: 16px;
  background: #FEF2F2 !important;
  border: 1.5px solid #FECACA !important;
  border-radius: 14px;
  padding: 1.2rem 1.6rem;
  margin-top: 1rem;
}
.emergency-icon-wrap { font-size: 2.4rem; }
.emergency-info { flex: 1; }
.emergency-title-row { display: flex; align-items: center; gap: 12px; }
.emergency-title-row h2 { font-size: 1.35rem; margin: 0; color: #DC2626 !important; font-weight: 800; }
.service-desc { color: #334155 !important; font-size: 1rem; font-weight: 600; margin: 4px 0 0; }
.badge { padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 800; }
.badge-critical { background: #DC2626; color: #FFF !important; }
.badge-urgent { background: #D97706; color: #FFF !important; }

.call-action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #16A34A;
  color: #FFFFFF !important;
  text-decoration: none !important;
  font-size: 1.2rem;
  font-weight: 800;
  padding: 14px 20px;
  border-radius: 12px;
  margin-top: 1.2rem;
}

.dir-container { margin-top: 1.8rem; }
.dir-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 1rem; color: #0F172A; }
.dir-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.dir-item { padding: 1rem 1.2rem; display: flex; align-items: center; gap: 14px; }
.dir-icon { font-size: 1.8rem; }
.dir-meta { flex: 1; }
.dir-meta h4 { margin: 0; font-size: 1rem; font-weight: 700; color: #0F172A; }
.dir-number { color: #0284C7 !important; font-weight: 800; font-size: 1.1rem; }
.quick-call-btn {
  background: #E0F2FE;
  color: #0284C7 !important;
  padding: 8px 16px;
  border-radius: 10px;
  text-decoration: none !important;
  font-weight: 700;
}
"""

INIT_JS = """() => {
  window._sosCtx = null;
  window._sosOsc = null;
  window._sosPlaying = false;
  window.startSiren = function() {
    if (window._sosPlaying) { window.stopSiren(); return; }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const gain = ctx.createGain();
    gain.gain.value = 0.9;
    gain.connect(ctx.destination);
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = 800;
    osc.connect(gain);
    osc.start();
    function sweep() {
      if (!window._sosPlaying) return;
      const now = ctx.currentTime;
      osc.frequency.setValueAtTime(800, now);
      osc.frequency.linearRampToValueAtTime(1600, now + 0.5);
      osc.frequency.linearRampToValueAtTime(800, now + 1.0);
      setTimeout(sweep, 1000);
    }
    window._sosCtx = ctx; window._sosOsc = osc; window._sosPlaying = true;
    sweep();
    setTimeout(() => { window.stopSiren(); }, 15000);
  };
  window.stopSiren = function() {
    window._sosPlaying = false;
    try { window._sosOsc && window._sosOsc.stop(); } catch(e) {}
    try { window._sosCtx && window._sosCtx.close(); } catch(e) {}
    window._sosOsc = null; window._sosCtx = null;
  };
}"""

# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(title="Sahaya Emergency", theme=gr.themes.Base(), css=CSS, js=INIT_JS) as demo:
    header = gr.HTML(header_html(t0))
    gr.HTML(
        '<div class="sos-row">'
        '<button class="sos-panic-btn" onclick="startSiren(); setTimeout(()=>window.open(\'tel:1091\'), 500);">'
        '🚨 EMERGENCY SOS &bull; LOUD ALARM + DIAL 1091'
        '</button>'
        '</div>'
    )
    lang_dd = gr.Dropdown(
        choices=[(label, code) for code, label in LANGUAGES.items()],
        value="kn_IN",
        label="🌐 Language Selection / ಭಾಷೆ / भाषा / ꯃꯤꯇꯩꯂꯣꯟ",
    )

    # 1. Voice Record Area: Full-width for standard, uncompressed mic controls
    mic = gr.Audio(
        sources=["microphone"],
        type="filepath",
        format="wav",
        label=t0["mic_header"],
        elem_classes=["audio-full-container"],
    )

    # 2. Text Input Area
    textbox = gr.Textbox(label=t0["type_header"], placeholder=t0["type_placeholder"], lines=2)
    send_btn = gr.Button(t0["send"], variant="primary")

    status = gr.Markdown("", visible=False)
    transcript_out = gr.Markdown("", elem_classes=["live-bubble"])
    reply_out = gr.Markdown("", elem_classes=["live-bubble"])
    reply_audio = gr.Audio(label="🔊 Sahaya Voice Response", type="filepath", autoplay=True, interactive=False)
    card_out = gr.HTML("")
    dir_html = gr.HTML(directory_html(t0))

    result_outputs = [status, transcript_out, reply_out, reply_audio, card_out]

    mic.stop_recording(on_audio, inputs=[mic, lang_dd], outputs=result_outputs)
    send_btn.click(on_text, inputs=[textbox, lang_dd], outputs=result_outputs)
    textbox.submit(on_text, inputs=[textbox, lang_dd], outputs=result_outputs)
    lang_dd.change(apply_language, inputs=lang_dd, outputs=[header, mic, textbox, send_btn, dir_html])

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
