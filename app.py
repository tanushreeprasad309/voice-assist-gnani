"""
Sahaya Emergency Assistant (Gradio on Render)
Voice-first emergency triage and first aid for India.

Gnani AI stack:
  Prisma  (speech-to-text)   -> https://api.vachana.ai/stt/v3
  Timbre  (text-to-speech)   -> https://api.vachana.ai/api/v1/tts/inference
Groq (triage reasoning)      -> https://api.groq.com/openai/v1/chat/completions

Environment variables:
  GNANI_API_KEY   Prisma + Timbre key
  GROQ_API_KEY    Groq API key
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
]

# ---------------------------------------------------------------------------
# Languages & UI Translations (12 Indian Languages including Manipuri)
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    "en_IN": {
        "title": "Sahaya",
        "subtitle": "Stay Safe.",
        "mic_header": "🎙️ Tap to Start Listening",
        "type_header": "⌨️ Or Type Emergency Details",
        "type_placeholder": "e.g., Need an ambulance, snake bite, road accident...",
        "send": "Send Emergency Alert",
        "detected": "🗣️ Detected Emergency:",
        "action_header": "Recommended Action & First-Aid",
        "first_aid_label": "Immediate First-Aid Steps:",
        "call_btn": "Call {number} Immediately",
        "directory_header": "Emergency Services Directory",
        "helpline": "Helpline:",
        "audio_response": "🔊 Sahaya Voice Guidance:",
        "say_call": "Please stay calm and call {number} immediately.",
    },
    "kn_IN": {
        "title": "ಸಹಾಯ (Sahaya)",
        "subtitle": "ಸುರಕ್ಷಿತವಾಗಿರಿ.",
        "mic_header": "🎙️ ಮಾತನಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "type_header": "⌨️ ಅಥವಾ ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ",
        "type_placeholder": "ಉದಾ. ನನಗೆ ಹಾವು ಕಚ್ಚಿದೆ, ಆಂಬ್ಯುಲೆನ್ಸ್ ಬೇಕು...",
        "send": "ತುರ್ತು ಎಚ್ಚರಿಕೆ ಕಳುಹಿಸಿ",
        "detected": "🗣️ ಪತ್ತೆಯಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ:",
        "action_header": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಮತ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "first_aid_label": "ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "ತುರ್ತು ಸಹಾಯವಾಣಿ ಡೈರೆಕ್ಟರಿ",
        "helpline": "ಸಹಾಯವಾಣಿ:",
        "audio_response": "🔊 ಸಹಾಯ ಧ್ವನಿ ಮಾರ್ಗದರ್ಶನ:",
        "say_call": "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ.",
    },
    "hi_IN": {
        "title": "सहायता (Sahaya)",
        "subtitle": "सुरक्षित रहें।",
        "mic_header": "🎙️ बोलने के लिए टैप करें",
        "type_header": "⌨️ या आपातकालीन विवरण टाइप करें",
        "type_placeholder": "जैसे एम्बुलेंस चाहिए, सांप ने काटा है, दुर्घटना हुई है...",
        "send": "आपातकालीन अलर्ट भेजें",
        "detected": "🗣️ पता चली आपात स्थिति:",
        "action_header": "अनुशंसित कार्रवाई और प्राथमिक उपचार",
        "first_aid_label": "प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "तुरंत {number} पर कॉल करें",
        "directory_header": "आपातकालीन सेवा डायरेक्टरी",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 सहायता वॉयस मार्गदर्शन:",
        "say_call": "कृपया शांत रहें और तुरंत {number} पर कॉल करें।",
    },
    "ta_IN": {
        "title": "சஹாயா (Sahaya)",
        "subtitle": "பாதுகாப்பாக இருங்கள்.",
        "mic_header": "🎙️ பேச தட்டவும்",
        "type_header": "⌨️ அல்லது தட்டச்சு செய்யவும்",
        "type_placeholder": "எ.கா. ஆம்புலன்ஸ் தேவை, பாம்பு கடித்தது, விபத்து...",
        "send": "அவசர எச்சரிக்கை அனுப்பு",
        "detected": "🗣️ கண்டறியப்பட்ட அவசரம்:",
        "action_header": "பரிந்துரைக்கப்பட்ட முதலுதவி",
        "first_aid_label": "முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "அவசர சேவை அடைவு",
        "helpline": "உதவி எண்:",
        "audio_response": "🔊 சஹாயா குரல் வழிகாட்டல்:",
        "say_call": "தயவுசெய்து உடனே {number} ஐ அழைக்கவும்.",
    },
    "te_IN": {
        "title": "సహాయ (Sahaya)",
        "subtitle": "సురక్షితంగా ఉండండి.",
        "mic_header": "🎙️ మాట్లాడటానికి నొక్కండి",
        "type_header": "⌨️ లేదా టైప్ చేయండి",
        "type_placeholder": "ఉదా. ఆంబులెన్స్ కావాలి, పాము కరిచింది, ప్రమాదం...",
        "send": "అత్యవసర హెచ్చరిక పంపండి",
        "detected": "🗣️ గుర్తించబడిన అత్యవసరం:",
        "action_header": "సిఫార్సు చేయబడిన ప్రథమ చికిత్స",
        "first_aid_label": "ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "అత్యవసర సేవల డైరెక్టరీ",
        "helpline": "హెల్ప్‌లైన్:",
        "audio_response": "🔊 సహాయ వాయిస్ మార్గదర్శకత్వం:",
        "say_call": "దయచేసి వెంటనే {number} కి కాల్ చేయండి.",
    },
    "ml_IN": {
        "title": "സഹായ (Sahaya)",
        "subtitle": "സുരക്ഷിതരായിരിക്കൂ.",
        "mic_header": "🎙️ സംസാരിക്കാൻ ടാപ്പ് ചെയ്യുക",
        "type_header": "⌨️ അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യുക",
        "type_placeholder": "ഉദാ. ആംബുലൻസ് വേണം, പാമ്പ് കടിയേറ്റു, അപകടം...",
        "send": "അടിയന്തര മുന്നറിയിപ്പ് അയയ്ക്കുക",
        "detected": "🗣️ കണ്ടെത്തിയ അത്യാഹിതം:",
        "action_header": "ശുപാർശ ചെയ്യുന്ന പ്രഥമശുശ്രൂഷ",
        "first_aid_label": "പ്രഥമശുശ്രൂഷ നിർദ്ദേശങ്ങൾ:",
        "call_btn": "ഉടൻ {number}-ലേക്ക് വിളിക്കുക",
        "directory_header": "എമർജൻസി ഡയറക്ടറി",
        "helpline": "ഹെൽപ്പ്‌ലൈൻ:",
        "audio_response": "🔊 സഹായ വോയ്സ് പ്രതികരണം:",
        "say_call": "ദയവായി ഉടൻ {number}-ലേക്ക് വിളിക്കുക.",
    },
    "mr_IN": {
        "title": "साहाय्य (Sahaya)",
        "subtitle": "सुरक्षित राहा.",
        "mic_header": "🎙️ बोलण्यासाठी टॅप करा",
        "type_header": "⌨️ किंवा टाइप करा",
        "type_placeholder": "उदा. रुग्णवाहिका हवी आहे, साप चावला, अपघात...",
        "send": "आणीबाणी सूचना पाठवा",
        "detected": "🗣️ आढळलेली आणीबाणी:",
        "action_header": "शिफारस केलेले प्रथमोपचार",
        "first_aid_label": "प्रथमोपचार सूचना:",
        "call_btn": "आता {number} वर कॉल करा",
        "directory_header": "आणीबाणी निर्देशिका",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 साहाय्य व्हॉइस मार्गदर्शन:",
        "say_call": "कृपया ताबडतोब {number} वर कॉल करा.",
    },
    "gu_IN": {
        "title": "સહાય (Sahaya)",
        "subtitle": "સુરક્ષિત રહો.",
        "mic_header": "🎙️ બોલવા માટે ટેપ કરો",
        "type_header": "⌨️ અથવા ટાઇપ કરો",
        "type_placeholder": "દા.ત. એમ્બ્યુલન્સ જોઈએ છે, સાપ કરડ્યો, અકસ્માત...",
        "send": "કટોકટી ચેતવણી મોકલો",
        "detected": "🗣️ મળેલી કટોકટી:",
        "action_header": "પ્રાથમિક સારવાર માર્ગદર્શિકા",
        "first_aid_label": "પ્રાથમિક સારવાર પગલાં:",
        "call_btn": "હમણાં {number} પર કૉલ કરો",
        "directory_header": "ઇમરજન્સી ડિરેક્ટરી",
        "helpline": "હેલ્પલાઇન:",
        "audio_response": "🔊 સહાય વૉઇસ માર્ગદર્શન:",
        "say_call": "કૃપા કરીને તરત {number} પર કૉલ કરો.",
    },
    "bn_IN": {
        "title": "সহায় (Sahaya)",
        "subtitle": "সুরক্ষিত থাকুন।",
        "mic_header": "🎙️ কথা বলতে ট্যাপ করুন",
        "type_header": "⌨️ অথবা টাইপ করুন",
        "type_placeholder": "যেমন অ্যাম্বুলেন্স দরকার, সাপে কেটেছে, দুর্ঘটনা...",
        "send": "জরুরি সতর্কতা পাঠান",
        "detected": "🗣️ শনাক্ত জরুরি অবস্থা:",
        "action_header": "প্রাথমিক চিকিৎসার নির্দেশিকা",
        "first_aid_label": "প্রাথমিক চিকিৎসা পদক্ষেপ:",
        "call_btn": "এখনই {number}-এ কল করুন",
        "directory_header": "জরুরি ডিরেক্টরি",
        "helpline": "হেল্পলাইন:",
        "audio_response": "🔊 সহায় ভয়েস সহায়তা:",
        "say_call": "অনুগ্রহ করে এখনই {number}-এ কল করুন।",
    },
    "pa_IN": {
        "title": "ਸਹਾਇਤਾ (Sahaya)",
        "subtitle": "ਸੁਰੱਖਿਅਤ ਰਹੋ।",
        "mic_header": "🎙️ ਬੋਲਣ ਲਈ ਟੈਪ ਕਰੋ",
        "type_header": "⌨️ ਜਾਂ ਟਾਈਪ ਕਰੋ",
        "type_placeholder": "ਜਿਵੇਂ ਐਂਬੂਲੈਂਸ ਚਾਹੀਦੀ ਹੈ, ਸੱਪ ਨੇ ਡੰਗਿਆ, ਹਾਦਸਾ...",
        "send": "ਐਮਰਜੈਂਸੀ ਅਲਰਟ ਭੇਜੋ",
        "detected": "🗣️ ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ:",
        "action_header": "ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼",
        "first_aid_label": "ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਕਦਮ:",
        "call_btn": "ਹੁਣੇ {number} 'ਤੇ ਕਾਲ ਕਰੋ",
        "directory_header": "ਐਮਰਜੈਂਸੀ ਡਾਇਰੈਕਟਰੀ",
        "helpline": "ਹੈਲਪਲਾਈਨ:",
        "audio_response": "🔊 ਸਹਾਇਤਾ ਵੌਇਸ ਸੇਧ:",
        "say_call": "ਕਿਰਪਾ ਕਰਕੇ ਤੁਰੰਤ {number} 'ਤੇ ਕਾਲ ਕਰੋ।",
    },
    "or_IN": {
        "title": "ସହାୟ (Sahaya)",
        "subtitle": "ସୁରକ୍ଷିତ ରୁହନ୍ତୁ।",
        "mic_header": "🎙️ କହିବାକୁ ଟ୍ୟାପ୍ କରନ୍ତୁ",
        "type_header": "⌨️ କିମ୍ବା ଟାଇପ୍ କରନ୍ତୁ",
        "type_placeholder": "ଯେପରି ଆମ୍ବୁଲାନ୍ସ ଦରକାର, ସାପ କାମୁଡ଼ିଛି, ଦୁର୍ଘଟଣା...",
        "send": "ଜରୁରୀକାଳୀନ ଆଲର୍ଟ ପଠାନ୍ତୁ",
        "detected": "🗣️ ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି:",
        "action_header": "ପ୍ରଥମ ଚିକିତ୍ସା ନିର୍ଦ୍ଦେଶାବଳୀ",
        "first_aid_label": "ପ୍ରଥମ ଚିକିତ୍ସା ପଦକ୍ଷେପ:",
        "call_btn": "ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ",
        "directory_header": "ଜରୁରୀକାଳୀନ ଡାଇରେକ୍ଟୋରୀ",
        "helpline": "ହେଲ୍ପଲାଇନ୍:",
        "audio_response": "🔊 ସହାୟ ଭଏସ୍ ମାର୍ଗଦର୍ଶନ:",
        "say_call": "ଦୟାକରି ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ।",
    },
    "mni_IN": {
        "title": "ꯁꯍꯥꯌ (Sahaya)",
        "subtitle": "ꯉꯥꯛꯊꯣꯛꯆꯕꯤꯌꯨ।",
        "mic_header": "🎙️ ꯋꯥ ꯉꯥꯡꯅꯕ ꯇꯦꯞ ꯇꯧꯕꯤꯌꯨ",
        "type_header": "⌨️ ꯅꯠꯇ꯭ꯔꯒ ꯇꯥꯏꯞ ꯇꯧꯕꯤꯌꯨ",
        "type_placeholder": "ꯑꯦꯝꯕꯨꯂꯦꯟꯁ ꯃꯊꯧ ꯇꯥꯏ, ꯂꯤꯟꯅ ꯆꯤꯛꯈ꯭ꯔꯦ, ꯈꯨꯗꯣꯡꯊꯤꯕ...",
        "send": "ꯈꯨꯗꯣꯡꯊꯤꯕꯒꯤ ꯄꯥꯎ ꯊꯥꯕꯤꯌꯨ",
        "detected": "🗣️ ꯈꯪꯗꯣꯛꯂꯕ ꯈꯨꯗꯣꯡꯊꯤꯕ:",
        "action_header": "ꯑꯍꯥꯟꯕ ꯃꯇꯦꯡ (First Aid)",
        "first_aid_label": "ꯇꯧꯒꯗꯕ ꯊꯕꯛꯁꯤꯡ:",
        "call_btn": "{number} ꯗ ꯍꯧꯖꯤꯛꯃꯛ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ",
        "directory_header": "ꯑꯇꯣꯞꯄ ꯃꯇꯦꯡ ꯅꯝꯕꯔꯁꯤꯡ",
        "helpline": "ꯍꯦꯜꯞꯂꯥꯏꯟ:",
        "audio_response": "🔊 ꯁꯍꯥꯌꯒꯤ ꯄꯥꯎꯈꯨꯝ:",
        "say_call": "ꯆꯥꯅꯕꯤꯗꯨꯅ ꯑꯀꯤꯕ ꯄꯣꯛꯂꯣꯏꯗꯕꯅꯤ, ꯈꯨꯗꯛꯇ {number} ꯗ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ.",
    },
}

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
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN", "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

# ---------------------------------------------------------------------------
# Emergency Directory
# ---------------------------------------------------------------------------
EMERGENCY_DIRECTORY = [
    {
        "title": "Medical & Ambulance",
        "icon": "🚑",
        "number": "108",
        "keywords": [
            "ambulance", "hospital", "doctor", "accident", "heart", "injury",
            "medical", "bleeding", "unconscious", "faint", "snake", "bite",
            "poison", "breathing", "saanp", "saaf", "kata", "haavu", "kacchide",
            "pambu", "paamu", "lin", "chikpa", "ꯂꯤꯟ", "ꯑꯦꯝꯕꯨꯂꯦꯟꯁ", "ꯍꯣꯁꯄꯤꯇꯥꯜ",
            "ಹಾವು", "ಕಚ್ಚಿದೆ", "ಆಂಬ್ಯುಲೆನ್ಸ್", "सांप", "एम्बुलेंस",
        ],
        "desc": "Medical emergencies, snake bites, trauma & urgent patient dispatch.",
        "first_aid": "1. Check airway, breathing & pulse.\n2. Keep person still and calm.\n3. Apply direct pressure to bleeding wounds.",
    },
    {
        "title": "Police & Safety",
        "icon": "👮",
        "number": "112",
        "keywords": [
            "police", "attack", "crime", "theft", "danger", "fight", "robbery",
            "chor", "bachao", "ꯄꯨꯂꯤꯁ", "ಪೊಲೀಸ್", "पुलिस", "போலீஸ்", "పోలీసు",
        ],
        "desc": "Immediate law enforcement intervention and urgent safety protection.",
        "first_aid": "1. Move to a safe, visible public location.\n2. Keep dispatch informed of nearby landmarks.",
    },
    {
        "title": "Women Distress Helpline",
        "icon": "🛡️",
        "number": "1091",
        "keywords": [
            "assault", "harassment", "abuse", "stalking", "threat", "molestation",
            "ಕಿರುಕುಳ", "दबंग", "उत्पीड़न", "துன்புறுத்தல்", "వేధింపులు",
        ],
        "desc": "Direct protection, helpline assistance and intervention against harassment or assault.",
        "first_aid": "1. Seek immediate public shelter.\n2. Dial 1091 or 112 instantly.\n3. Avoid confrontation alone.",
    },
    {
        "title": "Fire & Rescue",
        "icon": "🚒",
        "number": "101",
        "keywords": ["fire", "burn", "smoke", "cylinder", "blast", "gas", "aag", "benki", "ꯃꯩ", "ಬೆಂಕಿ", "आग", "തീ", "தீ"],
        "desc": "Fire hazard suppression, burn mitigation and urgent structure rescue.",
        "first_aid": "1. Evacuate immediately.\n2. Cool minor burns with clean tap water for 10-15 minutes.",
    },
    {
        "title": "Disaster Support",
        "icon": "🌊",
        "number": "1078",
        "keywords": ["flood", "earthquake", "cyclone", "landslide", "disaster", "baadh", "ꯏꯁꯤꯡ", "ಪ್ರವಾಹ", "बाढ़"],
        "desc": "Relief and evacuation assistance during floods, landslides and severe storms.",
        "first_aid": "1. Disconnect main power lines.\n2. Move calmly to higher concrete ground.",
    },
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}

# ---------------------------------------------------------------------------
# Helpers & Detection
# ---------------------------------------------------------------------------
def get_env(name):
    return os.getenv(name, "").strip()

def to_hyphen(code):
    return code.replace("_", "-")

SCRIPT_RANGES = [
    (0x0900, 0x097F, "hi_IN"),
    (0x0980, 0x09FF, "bn_IN"),
    (0x0A00, 0x0A7F, "pa_IN"),
    (0x0A80, 0x0AFF, "gu_IN"),
    (0x0B00, 0x0B7F, "or_IN"),
    (0x0B80, 0x0BFF, "ta_IN"),
    (0x0C00, 0x0C7F, "te_IN"),
    (0x0C80, 0x0CFF, "kn_IN"),
    (0x0D00, 0x0D7F, "ml_IN"),
    (0xABC0, 0xABFF, "mni_IN"),  # Meitei Mayek
]

ROMAN_LANG_KEYWORDS = {
    "hi_IN": {"mujhe", "saanp", "saaf", "kata", "kaata", "bachao", "madad", "khoon", "dard", "aag", "chot", "hai", "karo"},
    "kn_IN": {"nanage", "haavu", "havu", "kacchide", "kachide", "beku", "sahaya", "rakta", "novu", "benki", "dayavittu"},
    "ta_IN": {"enakku", "pambu", "paambu", "kadichathu", "udavi", "vendum", "ratham", "vali", "thee", "kaapathunga"},
    "te_IN": {"naaku", "paamu", "karichindi", "sahayam", "kavali", "raktam", "noppi", "pramadam", "kapadandi"},
    "ml_IN": {"enikku", "paampu", "kadichu", "sahayam", "venam", "raktham", "vedana", "thee", "rakshikku"},
    "mr_IN": {"mala", "saap", "chavla", "madat", "havi", "rakta", "dukhata", "aag", "vachva"},
    "mni_IN": {"eikhoi", "lin", "chikre", "chikpa", "mateng", "pibiyu", "ee", "amashung", "thuna"},
}

def detect_lang(text, fallback):
    if not text:
        return fallback
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        best = max(counts, key=counts.get)
        if best == "hi_IN" and fallback == "mr_IN":
            return "mr_IN"
        return best

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    roman_scores = {}
    for code, kw_set in ROMAN_LANG_KEYWORDS.items():
        matched = words & kw_set
        if matched:
            roman_scores[code] = len(matched)
    if roman_scores:
        return max(roman_scores, key=roman_scores.get)

    return fallback

def match_emergency(text):
    if not text:
        return []
    lower = text.lower()
    return [s for s in EMERGENCY_DIRECTORY if any(kw in lower for kw in s["keywords"])]

def get_fallback_response(transcript, lang_code, service):
    fallbacks = {
        "kn_IN": (
            "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ. ತಕ್ಷಣ 108 ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ ಮತ್ತು ರೋಗಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ.",
            [
                "ವ್ಯಕ್ತಿಯ ಉಸಿರಾಟವನ್ನು ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಶಾಂತವಾಗಿ ಮಲಗಿಸಿ.",
                "ಗಾಯದ ಮೇಲೆ ಒತ್ತಡ ಹಾಕಿ ರಕ್ತಸ್ರಾವ ನಿಲ್ಲಿಸಿ.",
                "ತಕ್ಷಣ ತುರ್ತು ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಿರಿ.",
            ],
        ),
        "hi_IN": (
            "कृपया घबराएं नहीं। तुरंत 108 पर कॉल करें और व्यक्ति को स्थिर रखें।",
            [
                "व्यक्ति की सांस और होश की जांच करें।",
                "घाव पर साफ कपड़े से दबाव बनाकर रखें।",
                "तुरंत एम्बुलेंस की प्रतीक्षा करें।",
            ],
        ),
        "mni_IN": (
            "ꯆꯥꯅꯕꯤꯗꯨꯅ ꯑꯀꯤꯕ ꯄꯣꯛꯂꯣꯏꯗꯕꯅꯤ। ꯈꯨꯗꯛꯇ 108 ꯑꯦꯝꯕꯨꯂꯦꯟꯁ ꯀꯣꯜ ꯇꯧꯕꯤꯌꯨ।",
            [
                "ꯃꯤꯑꯣꯏ ꯑꯗꯨꯕꯨ ꯅꯤꯡꯊꯤꯅ ꯂꯩꯍꯟꯕꯤꯌꯨ।",
                "ꯈꯨꯗꯣꯡꯊꯤꯕ ꯃꯐꯝ ꯑꯗꯨ ꯍꯤꯡꯅ ꯊꯝꯃꯣꯏꯗꯕꯅꯤ।",
                "ꯍꯣꯁꯄꯤꯇꯥꯜꯗ ꯈꯨꯗꯛꯇ ꯄꯨꯈꯠꯄꯤꯌꯨ।",
            ],
        ),
        "en_IN": (
            "Please stay calm. Immediately call 108 ambulance and follow basic first-aid.",
            [
                "Keep the patient completely calm and still.",
                "Apply firm pressure to bleeding injuries.",
                "Wait for medical dispatch.",
            ],
        ),
    }
    return fallbacks.get(lang_code, fallbacks.get("en_IN"))

# ---------------------------------------------------------------------------
# Gnani Prisma (STT) & Groq Triage
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

def groq_triage(transcript, lang_code):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "Groq key missing"

    language_name = LANG_NAMES.get(lang_code, "English")
    system_prompt = (
        "You are Sahaya, an empathetic AI emergency first-aid dispatcher.\n"
        f"1. Target Language: {language_name} ({lang_code}). Detect Romanized/transliterated Indian speech if used.\n"
        "2. Script: Respond ENTIRELY in the native script of the detected Indian language "
        "(e.g., Devanagari for Hindi, Kannada script for Kannada, Meitei Mayek for Manipuri, etc.). "
        "Do not answer in English if an Indian language is spoken.\n"
        "3. Output valid JSON ONLY:\n"
        '{"lang_code": "' + lang_code + '", '
        '"category": "medical|police|assault|fire|disaster", '
        '"urgency": "critical|urgent|routine", '
        '"reply": "Calming 2-sentence response in native script with urgent instruction", '
        '"steps": ["step 1 in native script", "step 2 in native script", "step 3 in native script"]}'
    )

    custom_model = get_env("GROQ_MODEL")
    models_to_try = ([custom_model] if custom_model else []) + [
        m for m in GROQ_MODELS if m != custom_model
    ]

    for model_name in models_to_try:
        try:
            r = requests.post(
                GROQ_CHAT_URL,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {groq_key}"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"User emergency: {transcript}"},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 550,
                },
                timeout=25,
            )
            if r.status_code != 200:
                continue

            content = r.json()["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end <= start:
                continue

            data = json.loads(content[start : end + 1])
            reply = str(data.get("reply", "")).strip()
            if not reply:
                continue

            steps = [re.sub(r"^\d+\.\s*", "", str(s).strip()) for s in data.get("steps", []) if str(s).strip()][:5]
            det_lang = str(data.get("lang_code", lang_code)).strip()
            if det_lang not in TRANSLATIONS:
                det_lang = lang_code

            return {
                "lang_code": det_lang,
                "category": str(data.get("category", "medical")).strip().lower(),
                "urgency": str(data.get("urgency", "urgent")).strip().lower(),
                "reply": reply,
                "steps": steps,
            }, ""
        except Exception:
            continue

    return None, "Offline fallback active"

# ---------------------------------------------------------------------------
# Gnani Timbre: TTS
# ---------------------------------------------------------------------------
def timbre_speak(text, lang_code):
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
            timeout=30,
        )
        if r.status_code != 200:
            return None, "TTS failed"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(r.content)
            return tmp.name, ""
    except Exception:
        return None, "TTS error"

# ---------------------------------------------------------------------------
# HTML Builders (Warm Sand/Beige Theme)
# ---------------------------------------------------------------------------
def header_html(t):
    return (
        '<div class="brand-top">'
        '  <div class="brand-title-box">'
        f'   <h1>{html.escape(t["title"])}</h1>'
        '    <div class="audio-wave"><span></span><span></span><span></span><span></span></div>'
        '  </div>'
        f' <p class="brand-sub">{html.escape(t["subtitle"])}</p>'
        '</div>'
    )

def result_card(service, steps, urgency, rt):
    if steps:
        body = "<ol class='step-list'>" + "".join(f"<li>{html.escape(s)}</li>" for s in steps) + "</ol>"
    else:
        body = "<p>" + html.escape(service["first_aid"]).replace("\n", "<br>") + "</p>"

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
        f' <div class="card-heading">🩹 {html.escape(rt["first_aid_label"])}</div>'
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
# Execution Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(transcript, selected_lang):
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
        matches = match_emergency(transcript)
        service = matches[0] if matches else EMERGENCY_DIRECTORY[0]
        reply_text, steps = get_fallback_response(transcript, reply_lang, service)
        urgency = "urgent"

    reply_bubble = f"**{rt['audio_response']}**\n\n{reply_text}"
    card = result_card(service, steps, urgency, rt)
    yield "", transcript_bubble, reply_bubble, None, card

    audio_path, _ = timbre_speak(reply_text, reply_lang)
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
# CSS Styling (Warm Sand / Ochre Theme & Black Header / Label Override)
# ---------------------------------------------------------------------------
CSS = """
:root {
  --body-background-fill: #EFE4CC !important;
  --background-fill-primary: #F8F1DF !important;
  --background-fill-secondary: #E9DEBE !important;
  --block-background-fill: #F8F1DF !important;
  --block-border-color: #D3C39E !important;
  --input-background-fill: #FAF6EA !important;
  --input-border-color: #C8B892 !important;
  --body-text-color: #2D2214 !important;
  --block-label-text-color: #3C2E1D !important;
}

body, .gradio-container, .dark, body.dark {
  background-color: #EFE4CC !important;
  color: #2D2214 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  max-width: 900px !important;
  margin: 0 auto !important;
  padding: 1.5rem 1rem !important;
}

footer { display: none !important; }

/* Force light warm background on all cards and blocks */
.gradio-container .block, 
.gradio-container .gr-panel, 
.gradio-container .gr-box,
.card-box {
  background: #F8F1DF !important;
  border: 1.5px solid #D7C7A3 !important;
  border-radius: 18px !important;
  box-shadow: 0 4px 16px rgba(60, 40, 15, 0.05) !important;
}

/* Remove dark/black background from component labels & headers */
.gradio-container .block-title,
.gradio-container .label,
.gradio-container label,
.gradio-container span.label-text,
.gradio-container [data-testid="block-label"] {
  background: transparent !important;
  background-color: transparent !important;
  color: #2D2214 !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
}

h1, h2, h3, h4, strong { color: #23190E !important; }
p, span { color: #4A3A25 !important; }

/* Header area */
.brand-top {
  text-align: center;
  padding: 1rem 1rem 0.5rem;
}

.brand-title-box {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.brand-top h1 {
  font-size: 2.4rem;
  font-weight: 800;
  margin: 0;
  color: #23190E !important;
  letter-spacing: -0.02em;
}

.brand-sub {
  font-size: 1rem;
  font-weight: 600;
  color: #6C5A40 !important;
  margin-top: 4px;
}

/* Wave icon next to Sahaya */
.audio-wave {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  height: 22px;
}
.audio-wave span {
  width: 4px;
  background: #2D2214;
  border-radius: 4px;
}
.audio-wave span:nth-child(1) { height: 10px; }
.audio-wave span:nth-child(2) { height: 20px; }
.audio-wave span:nth-child(3) { height: 14px; }
.audio-wave span:nth-child(4) { height: 8px; }

/* Panic SOS button */
.sos-row { 
  text-align: center; 
  margin: 0.5rem 0 1.2rem; 
}

.sos-panic-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  background: linear-gradient(135deg, #E03131 0%, #B01E1E 100%) !important;
  color: #FFFFFF !important;
  font-size: 1.05rem;
  font-weight: 700;
  padding: 14px 34px;
  border-radius: 50px;
  border: none !important;
  box-shadow: 0 8px 20px rgba(176, 30, 30, 0.3);
  cursor: pointer;
  transition: transform 0.15s ease;
}

.sos-panic-btn:hover {
  transform: translateY(-2px);
}

/* Clean Form inputs: no black pill backgrounds */
.gradio-container select, 
.gradio-container input, 
.gradio-container textarea,
.gradio-container .gr-input,
.gradio-container .secondary-wrap {
  background: #FAF6EA !important;
  border: 1.5px solid #C8B892 !important;
  border-radius: 14px !important;
  color: #2D2214 !important;
  font-size: 1rem !important;
}

/* Microphone & Audio player styling */
.gradio-container audio,
.gradio-container .gr-audio,
.gradio-container .audio-container,
.gradio-container [data-testid="audio"] {
  background: #F8F1DF !important;
  border: 1px solid #D7C7A3 !important;
  border-radius: 14px !important;
}

.gradio-container button.primary {
  background: #1D6AE5 !important;
  color: #FFFFFF !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 24px !important;
  box-shadow: 0 4px 14px rgba(29, 106, 229, 0.25) !important;
}

/* Response text cards */
.live-bubble {
  background: #FAF6EA !important;
  border: 1.5px solid #D7C7A3 !important;
  border-radius: 14px !important;
  padding: 16px 20px !important;
  margin: 10px 0 !important;
  color: #2D2214 !important;
}

/* Emergency banner */
.emergency-banner {
  display: flex;
  align-items: center;
  gap: 16px;
  background: #FCE8E6 !important;
  border: 1.5px solid #F5B5B0 !important;
  border-radius: 16px;
  padding: 1.2rem 1.6rem;
  margin-top: 1rem;
}

.emergency-icon-wrap {
  font-size: 2.2rem;
  background: #FFFFFF;
  width: 54px;
  height: 54px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(220, 38, 38, 0.12);
}

.emergency-info { flex: 1; }

.emergency-title-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.emergency-title-row h2 {
  font-size: 1.35rem;
  margin: 0;
  color: #C5221F !important;
  font-weight: 800;
}

.service-desc {
  color: #5C4730 !important;
  margin: 4px 0 0;
  font-size: 0.95rem;
}

.badge {
  padding: 3px 12px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 800;
}
.badge-critical { background: #DC2626; color: #FFF !important; }
.badge-urgent { background: #D97706; color: #FFF !important; }
.badge-routine { background: #059669; color: #FFF !important; }

/* Content card & buttons */
.content-card {
  padding: 1.5rem 1.8rem;
  margin-top: 1rem;
}

.card-heading {
  font-size: 1.15rem;
  font-weight: 700;
  color: #2D2214;
  margin-bottom: 0.8rem;
}

.step-list {
  margin: 0;
  padding-left: 1.4rem;
  color: #2D2214;
  line-height: 1.8;
}

.call-action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #15803D;
  color: #FFFFFF !important;
  text-decoration: none !important;
  font-size: 1.15rem;
  font-weight: 700;
  padding: 14px 20px;
  border-radius: 12px;
  margin-top: 1.3rem;
  box-shadow: 0 4px 14px rgba(21, 128, 61, 0.25);
}

/* Directory */
.dir-container { margin-top: 1.8rem; }
.dir-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 1rem; color: #2D2214; }
.dir-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}

.dir-item {
  padding: 1rem 1.3rem;
  display: flex;
  align-items: center;
  gap: 14px;
}

.dir-icon { font-size: 1.8rem; }
.dir-meta { flex: 1; }
.dir-meta h4 { margin: 0; font-size: 1rem; font-weight: 700; color: #2D2214 !important; }
.dir-number { color: #1D6AE5 !important; font-weight: 800; font-size: 0.95rem; }

.quick-call-btn {
  background: rgba(29, 106, 229, 0.1);
  color: #1D6AE5 !important;
  padding: 8px 16px;
  border-radius: 10px;
  text-decoration: none !important;
  font-weight: 700;
  font-size: 0.9rem;
  border: 1px solid rgba(29, 106, 229, 0.2);
}
"""

INIT_JS = """() => {
  window._sosCtx = null;
  window._sosOsc = null;
  window._sosPlaying = false;

  window.startSiren = function() {
    if (window._sosPlaying) {
      window.stopSiren();
      return;
    }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const gain = ctx.createGain();
    gain.gain.value = 1.0;
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

    window._sosCtx = ctx;
    window._sosOsc = osc;
    window._sosPlaying = true;
    sweep();

    setTimeout(function() {
      window.stopSiren();
    }, 15000);
  };

  window.stopSiren = function() {
    window._sosPlaying = false;
    try { window._sosOsc && window._sosOsc.stop(); } catch(e) {}
    try { window._sosCtx && window._sosCtx.close(); } catch(e) {}
    window._sosOsc = null;
    window._sosCtx = null;
  };
}"""

# ---------------------------------------------------------------------------
# Gradio Layout
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(
    title="Sahaya Emergency",
    theme=gr.themes.Base(),
    css=CSS,
    js=INIT_JS,
) as demo:
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
        value="en_IN",
        label="🌐 Language Selection / ಭಾಷೆ / भाषा / ꯃꯤꯇꯩꯂꯣꯟ",
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                format="wav",
                label=t0["mic_header"],
            )
        with gr.Column():
            textbox = gr.Textbox(
                label=t0["type_header"],
                placeholder=t0["type_placeholder"],
                lines=3,
            )
            send_btn = gr.Button(t0["send"], variant="primary")

    status = gr.Markdown("", visible=False)
    transcript_out = gr.Markdown("", elem_classes=["live-bubble"])
    reply_out = gr.Markdown("", elem_classes=["live-bubble"])
    reply_audio = gr.Audio(
        label="🔊 Sahaya Voice Response",
        type="filepath",
        autoplay=True,
        interactive=False,
    )
    card_out = gr.HTML("")
    dir_html = gr.HTML(directory_html(t0))

    result_outputs = [status, transcript_out, reply_out, reply_audio, card_out]

    mic.stop_recording(on_audio, inputs=[mic, lang_dd], outputs=result_outputs)
    send_btn.click(on_text, inputs=[textbox, lang_dd], outputs=result_outputs)
    textbox.submit(on_text, inputs=[textbox, lang_dd], outputs=result_outputs)

    lang_dd.change(
        apply_language,
        inputs=lang_dd,
        outputs=[header, mic, textbox, send_btn, dir_html],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
