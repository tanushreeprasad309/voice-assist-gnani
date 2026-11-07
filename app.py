"""
Sahaya Emergency Assistant (Gradio on Render)
Voice-first emergency triage and first aid for rural India.

Gnani AI stack:
  Prisma  (speech-to-text)   -> https://api.vachana.ai/stt/v3
  Timbre  (text-to-speech)   -> https://api.vachana.ai/api/v1/tts/inference
Groq (free triage reasoning) -> https://api.groq.com/openai/v1

Environment variables to set in Render Dashboard -> Environment:
  GNANI_API_KEY   Prisma + Timbre key
  GROQ_API_KEY    Free key from console.groq.com
Optional:
  TIMBRE_MODEL    default timbre-v3.0   (or timbre-v2.5)
  TIMBRE_VOICE    default Shivani       (must belong to the chosen model)
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

# Groq settings (free tier)
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"   # free on Groq

# ---------------------------------------------------------------------------
# 11-language UI dictionary, language lists and emergency directory
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    "en_IN": {
        "title": "🚨 Sahaya Emergency Assistant",
        "subtitle": "Multilingual Voice First-Aid & Emergency Dispatch",
        "mic_header": "🎙️ Speak Your Emergency",
        "mic_start": "🔴 Tap to Speak",
        "mic_stop": "⏹️ Stop",
        "type_header": "⌨️ Or Type It",
        "type_placeholder": "e.g. Need an ambulance, there has been an accident",
        "send": "Send",
        "detected": "🗣️ **Detected Emergency:**",
        "action_header": "⚡ Recommended Action & First-Aid",
        "first_aid_label": "🩹 First-Aid Guidelines:",
        "call_btn": "📞 Call {number} Now",
        "directory_header": "📋 Emergency Directory (Tap Any Card to Call)",
        "helpline": "Helpline:",
        "audio_response": "🔊 **Sahaya's Spoken Response:**",
        "say_call": "Please call {number} immediately.",
    },
    "kn_IN": {
        "title": "🚨 ಸಹಾಯ ತುರ್ತು ಸಹಾಯಕ",
        "subtitle": "ಬಹುಭಾಷಾ ಧ್ವನಿ ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ ಮತ್ತು ತುರ್ತು ಪ್ರಸರಣ",
        "mic_header": "🎙️ ನಿಮ್ಮ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯನ್ನು ಹೇಳಿ",
        "mic_start": "🔴 ಮಾತನಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "mic_stop": "⏹️ ನಿಲ್ಲಿಸಿ",
        "type_header": "⌨️ ಅಥವಾ ಟೈಪ್ ಮಾಡಿ",
        "type_placeholder": "ಉದಾ. ಆಂಬ್ಯುಲೆನ್ಸ್ ಬೇಕು ಅಥವಾ ಅಪಘಾತವಾಗಿದೆ",
        "send": "ಕಳುಹಿಸಿ",
        "detected": "🗣️ **ಪತ್ತೆಯಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ:**",
        "action_header": "⚡ ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಮತ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "first_aid_label": "🩹 ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಮಾರ್ಗಸೂಚಿಗಳು:",
        "call_btn": "📞 ಈಗ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "📋 ತುರ್ತು ಡೈರೆಕ್ಟರಿ (ಕರೆ ಮಾಡಲು ಕಾರ್ಡ್ ಅನ್ನು ಟ್ಯಾಪ್ ಮಾಡಿ)",
        "helpline": "ಸಹಾಯವಾಣಿ:",
        "audio_response": "🔊 **ಸಹಾಯ ಧ್ವನಿ ಪ್ರತಿಕ್ರಿಯೆ:**",
        "say_call": "ದಯವಿಟ್ಟು ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ.",
    },
    "hi_IN": {
        "title": "🚨 सहायता आपातकालीन सहायक",
        "subtitle": "बहुभाषी वॉयस फर्स्ट-एड और आपातकालीन प्रेषण",
        "mic_header": "🎙️ अपनी आपात स्थिति बोलें",
        "mic_start": "🔴 बोलने के लिए टैप करें",
        "mic_stop": "⏹️ रोकें",
        "type_header": "⌨️ या टाइप करें",
        "type_placeholder": "जैसे एम्बुलेंस चाहिए, दुर्घटना हुई है",
        "send": "भेजें",
        "detected": "🗣️ **पता चली आपात स्थिति:**",
        "action_header": "⚡ अनुशंसित कार्रवाई और प्राथमिक उपचार",
        "first_aid_label": "🩹 प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "📞 अभी {number} पर कॉल करें",
        "directory_header": "📋 आपातकालीन डायरेक्टरी (कॉल करने के लिए टैप करें)",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 **सहायता की मौखिक प्रतिक्रिया:**",
        "say_call": "कृपया तुरंत {number} पर कॉल करें।",
    },
    "ta_IN": {
        "title": "🚨 சஹாயா அவசர உதவி",
        "subtitle": "பன்மொழி குரல் முதலுதவி & அவசர அனுப்புதல்",
        "mic_header": "🎙️ உங்கள் அவசரநிலையைச் சொல்லுங்கள்",
        "mic_start": "🔴 பேச தட்டவும்",
        "mic_stop": "⏹️ நிறுத்து",
        "type_header": "⌨️ அல்லது தட்டச்சு செய்யவும்",
        "type_placeholder": "எ.கா. ஆம்புலன்ஸ் தேவை, விபத்து நடந்தது",
        "send": "அனுப்பு",
        "detected": "🗣️ **கண்டறியப்பட்ட அவசரம்:**",
        "action_header": "⚡ பரிந்துரைக்கப்பட்ட நடவடிக்கை மற்றும் முதலுதவி",
        "first_aid_label": "🩹 முதலுதவி வழிகாட்டுதல்கள்:",
        "call_btn": "📞 உடனே {number} ஐ அழைக்கவும்",
        "directory_header": "📋 அவசர அடைவு (அழைக்க தட்டவும்)",
        "helpline": "உதவி எண்:",
        "audio_response": "🔊 **சஹாயாவின் குரல் பதில்:**",
        "say_call": "தயவுசெய்து உடனே {number} ஐ அழைக்கவும்.",
    },
    "te_IN": {
        "title": "🚨 సహాయ అత్యవసర సహాయకుడు",
        "subtitle": "బహుభాషా వాయిస్ ఫస్ట్-ఎయిడ్ & ఎమర్జెన్సీ డిస్పాచ్",
        "mic_header": "🎙️ మీ అత్యవసర పరిస్థితిని చెప్పండి",
        "mic_start": "🔴 మాట్లాడటానికి నొక్కండి",
        "mic_stop": "⏹️ ఆపు",
        "type_header": "⌨️ లేదా టైప్ చేయండి",
        "type_placeholder": "ఉదా. ఆంబులెన్స్ కావాలి, ప్రమాదం జరిగింది",
        "send": "పంపండి",
        "detected": "🗣️ **గుర్తించబడిన అత్యవసరం:**",
        "action_header": "⚡ సిఫార్సు చేయబడిన చర్య మరియు ప్రథమ చికిత్స",
        "first_aid_label": "🩹 ప్రథమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "📞 ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "📋 అత్యవసర డైరెక్టరీ (కాల్ చేయడానికి ట్యాప్ చేయండి)",
        "helpline": "హెల్ప్‌లైన్:",
        "audio_response": "🔊 **సహాయ వాయిస్ ప్రతిస్పందన:**",
        "say_call": "దయచేసి వెంటనే {number} కి కాల్ చేయండి.",
    },
    "ml_IN": {
        "title": "🚨 സഹായ അടിയന്തര അസിസ്റ്റന്റ്",
        "subtitle": "ബഹുഭാഷാ വോയ്സ് ഫസ്റ്റ് എയ്ഡ് & എമർജൻസി ഡിസ്പാച്ച്",
        "mic_header": "🎙️ നിങ്ങളുടെ അടിയന്തര സാഹചര്യം പറയൂ",
        "mic_start": "🔴 സംസാരിക്കാൻ ടാപ്പ് ചെയ്യുക",
        "mic_stop": "⏹️ നിർത്തുക",
        "type_header": "⌨️ അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യുക",
        "type_placeholder": "ഉദാ. ആംബുലൻസ് വേണം, അപകടം ഉണ്ടായി",
        "send": "അയയ്ക്കുക",
        "detected": "🗣️ **കണ്ടെത്തിയ അത്യാഹിതം:**",
        "action_header": "⚡ ശുപാർശ ചെയ്യുന്ന നടപടിയും പ്രഥമശുശ്രൂഷയും",
        "first_aid_label": "🩹 പ്രഥമശുശ്രൂഷ മാർഗ്ഗനിർദ്ദേശങ്ങൾ:",
        "call_btn": "📞 ഇപ്പോൾ {number}-ലേക്ക് വിളിക്കുക",
        "directory_header": "📋 എമർജൻസി ഡയറക്ടറി (വിളിക്കാൻ ടാപ്പ് ചെയ്യുക)",
        "helpline": "ഹെൽപ്പ്‌ലൈൻ:",
        "audio_response": "🔊 **സഹായയുടെ ശബ്ദ പ്രതികരണം:**",
        "say_call": "ദയവായി ഉടൻ {number}-ലേക്ക് വിളിക്കുക.",
    },
    "mr_IN": {
        "title": "🚨 साहाय्य आणीबाणी सहाय्यक",
        "subtitle": "बहुभाषिक व्हॉइस फर्स्ट-एड आणि आपत्कालीन प्रेषण",
        "mic_header": "🎙️ तुमची आणीबाणी सांगा",
        "mic_start": "🔴 बोलण्यासाठी टॅप करा",
        "mic_stop": "⏹️ थांबवा",
        "type_header": "⌨️ किंवा टाइप करा",
        "type_placeholder": "उदा. रुग्णवाहिका हवी आहे, अपघात झाला",
        "send": "पाठवा",
        "detected": "🗣️ **आढळलेली आणीबाणी:**",
        "action_header": "⚡ शिफारस केलेली कृती आणि प्रथमोपचार",
        "first_aid_label": "🩹 प्रथमोपचार मार्गदर्शक तत्त्वे:",
        "call_btn": "📞 आता {number} वर कॉल करा",
        "directory_header": "📋 आणीबाणी निर्देशिका (कॉल करण्यासाठी टॅप करा)",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 **साहाय्यचा बोललेला प्रतिसाद:**",
        "say_call": "कृपया ताबडतोब {number} वर कॉल करा.",
    },
    "gu_IN": {
        "title": "🚨 સહાય ઇમરજન્સી અસિસ્ટન્ટ",
        "subtitle": "બહુભાષી વૉઇસ ફર્સ્ટ-એઇડ અને ઇમરજન્સી ડિસ્પેચ",
        "mic_header": "🎙️ તમારી કટોકટી બોલો",
        "mic_start": "🔴 બોલવા માટે ટેપ કરો",
        "mic_stop": "⏹️ બંધ કરો",
        "type_header": "⌨️ અથવા ટાઇપ કરો",
        "type_placeholder": "દા.ત. એમ્બ્યુલન્સ જોઈએ છે, અકસ્માત થયો છે",
        "send": "મોકલો",
        "detected": "🗣️ **મળેલી કટોકટી:**",
        "action_header": "⚡ ભલામણ કરેલ પગલાં અને પ્રાથમિક સારવાર",
        "first_aid_label": "🩹 પ્રાથમિક સારવાર માર્ગદર્શિકા:",
        "call_btn": "📞 હમણાં જ {number} પર કૉલ કરો",
        "directory_header": "📋 ઇમરજન્સી ડિરેક્ટરી (કૉલ કરવા માટે ટેપ કરો)",
        "helpline": "હેલ્પલાઇન:",
        "audio_response": "🔊 **સહાયનો બોલાયેલ જવાબ:**",
        "say_call": "કૃપા કરીને તરત {number} પર કૉલ કરો.",
    },
    "bn_IN": {
        "title": "🚨 সহায় জরুরি সহকারী",
        "subtitle": "বহুভাষিক ভয়েস ফার্স্ট-এইড এবং জরুরি প্রেরণ",
        "mic_header": "🎙️ আপনার জরুরি অবস্থা বলুন",
        "mic_start": "🔴 কথা বলতে ট্যাপ করুন",
        "mic_stop": "⏹️ থামুন",
        "type_header": "⌨️ অথবা টাইপ করুন",
        "type_placeholder": "যেমন অ্যাম্বুলেন্স দরকার, দুর্ঘটনা ঘটেছে",
        "send": "পাঠান",
        "detected": "🗣️ **শনাক্ত জরুরি অবস্থা:**",
        "action_header": "⚡ প্রস্তাবিত পদক্ষেপ এবং প্রাথমিক চিকিৎসা",
        "first_aid_label": "🩹 প্রাথমিক চিকিৎসার নির্দেশিকা:",
        "call_btn": "📞 এখনই {number}-এ কল করুন",
        "directory_header": "📋 জরুরি ডিরেক্টরি (কল করতে ট্যাপ করুন)",
        "helpline": "হেল্পলাইন:",
        "audio_response": "🔊 **সহায়ের মৌখিক প্রতিক্রিয়া:**",
        "say_call": "অনুগ্রহ করে এখনই {number}-এ কল করুন।",
    },
    "pa_IN": {
        "title": "🚨 ਸਹਾਇਤਾ ਐਮਰਜੈਂਸੀ ਸਹਾਇਕ",
        "subtitle": "ਬਹੁ-ਭਾਸ਼ੀ ਵੌਇਸ ਫਸਟ-ਏਡ ਅਤੇ ਐਮਰਜੈਂਸੀ ਡਿਸਪੈਚ",
        "mic_header": "🎙️ ਆਪਣੀ ਐਮਰਜੈਂਸੀ ਦੱਸੋ",
        "mic_start": "🔴 ਬੋਲਣ ਲਈ ਟੈਪ ਕਰੋ",
        "mic_stop": "⏹️ ਰੋਕੋ",
        "type_header": "⌨️ ਜਾਂ ਟਾਈਪ ਕਰੋ",
        "type_placeholder": "ਜਿਵੇਂ ਐਂਬੂਲੈਂਸ ਚਾਹੀਦੀ ਹੈ, ਹਾਦਸਾ ਹੋਇਆ ਹੈ",
        "send": "ਭੇਜੋ",
        "detected": "🗣️ **ਪਤਾ ਲੱਗੀ ਐਮਰਜੈਂਸੀ:**",
        "action_header": "⚡ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਕਾਰਵਾਈ ਅਤੇ ਮੁੱਢਲੀ ਸਹਾਇਤਾ",
        "first_aid_label": "🩹 ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼:",
        "call_btn": "📞 ਹੁਣੇ {number} 'ਤੇ ਕਾਲ ਕਰੋ",
        "directory_header": "📋 ਐਮਰਜੈਂਸੀ ਡਾਇਰੈਕਟਰੀ (ਕਾਲ ਕਰਨ ਲਈ ਟੈਪ ਕਰੋ)",
        "helpline": "ਹੈਲਪਲਾਈਨ:",
        "audio_response": "🔊 **ਸਹਾਇਤਾ ਦਾ ਬੋਲਿਆ ਜਵਾਬ:**",
        "say_call": "ਕਿਰਪਾ ਕਰਕੇ ਤੁਰੰਤ {number} 'ਤੇ ਕਾਲ ਕਰੋ।",
    },
    "or_IN": {
        "title": "🚨 ସହାୟ ଜରୁରୀକାଳୀନ ସହାୟକ",
        "subtitle": "ବହୁଭାଷୀ ଭଏସ୍ ଫାଷ୍ଟ-ଏଡ୍ ଏବଂ ଜରୁରୀକାଳୀନ ପ୍ରେରଣ",
        "mic_header": "🎙️ ଆପଣଙ୍କ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି କୁହନ୍ତୁ",
        "mic_start": "🔴 କହିବାକୁ ଟ୍ୟାପ୍ କରନ୍ତୁ",
        "mic_stop": "⏹️ ବନ୍ଦ କରନ୍ତୁ",
        "type_header": "⌨️ କିମ୍ବା ଟାଇପ୍ କରନ୍ତୁ",
        "type_placeholder": "ଯେପରି ଆମ୍ବୁଲାନ୍ସ ଆବଶ୍ୟକ, ଦୁର୍ଘଟଣା ହୋଇଛି",
        "send": "ପଠାନ୍ତୁ",
        "detected": "🗣️ **ଚିହ୍ନଟ ଜରୁରୀକାଳୀନ ପରିସ୍ଥିତି:**",
        "action_header": "⚡ ପରାମର୍ଶିତ କାର୍ଯ୍ୟ ଏବଂ ପ୍ରଥମ ଚିକିତ୍ସା",
        "first_aid_label": "🩹 ପ୍ରଥମ ଚିକିତ୍ସା ନିର୍ଦ୍ଦେଶାବଳୀ:",
        "call_btn": "📞 ବର୍ତ୍ତମାନ {number} କୁ କଲ୍ କରନ୍ତୁ",
        "directory_header": "📋 ଜରୁରୀକାଳୀନ ଡାଇରେକ୍ଟୋରୀ (କଲ୍ କରିବାକୁ ଟ୍ୟାପ୍ କରନ୍ତୁ)",
        "helpline": "ହେଲ୍ପଲାଇନ୍:",
        "audio_response": "🔊 **ସହାୟଙ୍କ କଥିତ ପ୍ରତିକ୍ରିୟା:**",
        "say_call": "ଦୟାକରି ତୁରନ୍ତ {number} କୁ କଲ୍ କରନ୍ତୁ।",
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
}

LANG_NAMES = {
    "en_IN": "English", "kn_IN": "Kannada", "hi_IN": "Hindi", "ta_IN": "Tamil",
    "te_IN": "Telugu", "ml_IN": "Malayalam", "mr_IN": "Marathi", "gu_IN": "Gujarati",
    "bn_IN": "Bengali", "pa_IN": "Punjabi", "or_IN": "Odia",
}

# Languages Timbre can speak (Odia is not in the Timbre language list).
TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

# ---------------------------------------------------------------------------
# Emergency directory (used for the final card and as a keyword fallback)
# ---------------------------------------------------------------------------
EMERGENCY_DIRECTORY = [
    {
        "title": "Medical & Ambulance",
        "icon": "🚑",
        "number": "108",
        "keywords": ["ambulance", "hospital", "doctor", "accident", "heart", "injury", "medical",
                     "bleeding", "unconscious", "faint", "snake", "breathing", "pregnan",
                     "ಆಂಬ್ಯುಲೆನ್ಸ್", "ಆಸ್ಪತ್ರೆ", "ಅಪಘಾತ", "एम्बुलेंस", "अस्पताल", "दुर्घटना",
                     "चोट", "एक्सीडेंट", "ആംബുലൻസ്", "அவசர ஊர்தி", "ஆம்புலன்ஸ்", "ఆంబులెన్స్"],
        "desc": "Immediate dispatch for medical crises, severe injuries, or emergency transport.",
        "first_aid": "1. Check airway, breathing & pulse.\n2. Apply firm pressure to bleeding wounds.\n3. Keep the person calm until 108 arrives.",
    },
    {
        "title": "Police & Safety",
        "icon": "👮",
        "number": "112",
        "keywords": ["police", "attack", "crime", "theft", "danger", "fight", "robbery",
                     "ಪೊಲೀಸ್", "पुलिस", "बचाओ", "പോലീസ്", "காவல்துறை", "పోలీసు", "ପୋଲିସ"],
        "desc": "Immediate assistance for crime, personal safety threats, or law enforcement.",
        "first_aid": "1. Move to a safe, well-lit public area.\n2. Stay on the line with 112 and give landmarks.",
    },
    {
        "title": "Assault & Distress (Women/Men)",
        "icon": "🛡️",
        "number": "1091",
        "keywords": ["assault", "harassment", "abuse", "stalking", "threat", "molestation",
                     "ದೌರ್ಜನ್ಯ", "ಕಿರುಕುಳ", "उत्पीड़न", "छेड़छाड़", "ഉപദ്രവം", "துன்புறுத்தல்",
                     "వేధింపులు", "ଅତ୍ୟାଚାର"],
        "desc": "Dedicated protection and rapid response for assault, harassment, or severe distress.",
        "first_aid": "1. Seek immediate shelter in a safe shop or public group.\n2. Call 1091 or 112 instantly.\n3. Avoid confronting the person alone.",
    },
    {
        "title": "Fire & Rescue",
        "icon": "🚒",
        "number": "101",
        "keywords": ["fire", "burn", "smoke", "cylinder", "blast", "rescue",
                     "ಬೆಂಕಿ", "आग", "തീ", "தீ", "అగ్ని", "ନିଆଁ"],
        "desc": "Emergency response for fires, gas leaks, burns, or structural entrapment.",
        "first_aid": "1. Evacuate immediately and stay low under smoke.\n2. Run cool water over minor burns for 10-15 minutes.",
    },
    {
        "title": "Disaster & Flood Relief",
        "icon": "🌊",
        "number": "1078",
        "keywords": ["flood", "earthquake", "cyclone", "landslide", "disaster",
                     "ಭೂಕಂಪ", "ಪ್ರವಾಹ", "बाढ़", "പ്രളയം", "வெள்ளம்", "వరద", "ବନ୍ୟା"],
        "desc": "Support during natural calamities, earthquakes, floods, and severe weather.",
        "first_aid": "1. Switch off main electrical power and gas lines.\n2. Move to higher ground and avoid downed wires.",
    },
]

CATEGORY_INDEX = {"medical": 0, "police": 1, "assault": 2, "fire": 3, "disaster": 4}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_env(name):
    return os.getenv(name, "").strip()


def to_hyphen(code):
    return code.replace("_", "-")


SCRIPT_RANGES = [
    (0x0900, 0x097F, "hi_IN"),  # Devanagari (Hindi / Marathi)
    (0x0980, 0x09FF, "bn_IN"),
    (0x0A00, 0x0A7F, "pa_IN"),
    (0x0A80, 0x0AFF, "gu_IN"),
    (0x0B00, 0x0B7F, "or_IN"),
    (0x0B80, 0x0BFF, "ta_IN"),
    (0x0C00, 0x0C7F, "te_IN"),
    (0x0C80, 0x0CFF, "kn_IN"),
    (0x0D00, 0x0D7F, "ml_IN"),
]


def detect_lang(text, fallback):
    """Detect the language of the transcript from its script."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return fallback  # Latin script (English / Hinglish): keep selected language
    best = max(counts, key=counts.get)
    if best == "hi_IN" and fallback == "mr_IN":
        return "mr_IN"
    return best


def match_emergency(text):
    if not text:
        return []
    lower = text.lower()
    return [s for s in EMERGENCY_DIRECTORY if any(kw in lower for kw in s["keywords"])]


# ---------------------------------------------------------------------------
# Gnani Prisma: speech to text
# ---------------------------------------------------------------------------
def transcribe(audio_path, lang_code):
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return "", "GNANI_API_KEY is missing. Add it in Render's Environment settings."
    try:
        with open(audio_path, "rb") as f:
            r = requests.post(
                STT_URL,
                headers={"X-API-Key-ID": api_key},
                files={"audio_file": ("voice.wav", f, "audio/wav")},
                data={"language_code": to_hyphen(lang_code), "format": "transcribe"},
                timeout=30,
            )
    except Exception as e:
        return "", f"Could not reach the speech service: {e}"
    if r.status_code != 200:
        return "", f"Speech recognition failed ({r.status_code}): {r.text[:200]}"
    try:
        data = r.json()
    except ValueError:
        return "", "Unexpected response from the speech service."
    transcript = str(data.get("transcript") or data.get("text") or "").strip()
    if not transcript:
        return "", "No speech detected. Please try again, closer to the microphone."
    return transcript, ""


# ---------------------------------------------------------------------------
# Groq triage: medical reasoning via free Groq API (replaces Evon)
# ---------------------------------------------------------------------------
def groq_triage(transcript, lang_code):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "GROQ_API_KEY is missing, so keyword matching was used."

    language = LANG_NAMES.get(lang_code, "English")
    system_prompt = (
        "You are Sahaya, a calm first-aid and emergency voice assistant for rural India. "
        f"Reply ONLY in {language}, in that language's own script, using very simple everyday words. "
        "The user may mix languages (for example Hinglish or Kanglish); still reply in "
        f"{language}. Never prescribe medicines and give only basic first-aid steps. "
        "If life may be at risk, the first sentence of the reply must tell the person to call "
        "one zero eight (ambulance) now. Write all numbers as spoken words, not digits. "
        "Return JSON only, with no extra text, in exactly this shape: "
        '{"category": "medical|police|assault|fire|disaster", '
        '"urgency": "critical|urgent|routine", '
        '"reply": "<at most 3 short spoken sentences>", '
        '"steps": ["<short first-aid step>", "<short first-aid step>", "<short first-aid step>"]}'
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_key}",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    try:
        r = requests.post(f"{GROQ_URL}/chat/completions", headers=headers, json=payload, timeout=45)
        if r.status_code != 200:
            return None, f"Groq error ({r.status_code}), so keyword matching was used."
        content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return None, f"Groq request failed ({e}), so keyword matching was used."

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None, "Groq reply was not valid JSON, so keyword matching was used."
    try:
        data = json.loads(content[start:end + 1])
    except ValueError:
        return None, "Groq reply was not valid JSON, so keyword matching was used."

    reply = str(data.get("reply", "")).strip()
    if not reply:
        return None, "Groq returned an empty reply, so keyword matching was used."

    raw_steps = data.get("steps")
    steps = []
    if isinstance(raw_steps, list):
        steps = [str(s).strip() for s in raw_steps if str(s).strip()][:4]

    return {
        "category": str(data.get("category", "medical")).strip().lower(),
        "urgency": str(data.get("urgency", "")).strip().lower(),
        "reply": reply,
        "steps": steps,
    }, ""


# ---------------------------------------------------------------------------
# Gnani Timbre: text to speech in the same language
# ---------------------------------------------------------------------------
def timbre_speak(text, lang_code):
    if lang_code not in TIMBRE_LANGS:
        return None, "Voice reply is not available for this language yet; the text reply is shown."
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return None, "GNANI_API_KEY is missing, so no voice reply was generated."

    model = get_env("TIMBRE_MODEL") or "timbre-v3.0"
    voice = get_env("TIMBRE_VOICE") or "Shivani"
    try:
        r = requests.post(
            TTS_URL,
            headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "voice": voice,
                "model": model,
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
    except Exception as e:
        return None, f"Could not reach the voice service: {e}"
    if r.status_code != 200:
        return None, f"Voice reply failed ({r.status_code}): {r.text[:200]}"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(r.content)
    return tmp.name, ""


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def call_link(number, label):
    return f'<a class="call-btn" href="tel:{number}">{html.escape(label)}</a>'


def header_html(t):
    return (
        f'<div class="header-banner"><h1>{html.escape(t["title"])}</h1>'
        f'<p>{html.escape(t["subtitle"])}</p></div>'
    )


def result_card(service, steps, urgency, rt):
    if steps:
        body = "<ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in steps) + "</ol>"
    else:
        body = "<br>" + html.escape(service["first_aid"]).replace("\n", "<br>")

    pill = ""
    if urgency in ("critical", "urgent", "routine"):
        pill = f'<span class="pill pill-{urgency}">{urgency.upper()}</span>'

    return "".join([
        '<div class="result-card">',
        f'<h3>{service["icon"]} {html.escape(service["title"])}{pill}</h3>',
        f'<p>{html.escape(service["desc"])}</p>',
        '<div class="step-box">',
        f'<strong>{html.escape(rt["first_aid_label"])}</strong>',
        body,
        "</div>",
        call_link(service["number"], rt["call_btn"].format(number=service["number"])),
        "</div>",
    ])


def directory_html(t):
    cards = []
    for srv in EMERGENCY_DIRECTORY:
        cards.append("".join([
            '<div class="grid-card"><div>',
            f'<h2>{srv["icon"]}</h2>',
            f'<h3>{html.escape(srv["title"])}</h3>',
            f'<p>{html.escape(srv["desc"])}</p>',
            "</div><div>",
            f'<div class="helpline">{html.escape(t["helpline"])} {srv["number"]}</div>',
            call_link(srv["number"], t["call_btn"].format(number=srv["number"])),
            "</div></div>",
        ]))
    return (
        f'<h3 class="dir-title">{html.escape(t["directory_header"])}</h3>'
        f'<div class="dir-grid">{"".join(cards)}</div>'
    )


# ---------------------------------------------------------------------------
# Pipeline (generator: results appear step by step, which feels real-time)
# ---------------------------------------------------------------------------
def run_pipeline(transcript, selected_lang):
    reply_lang = detect_lang(transcript, selected_lang)
    rt = TRANSLATIONS[reply_lang]
    transcript_md = f"{rt['detected']} {transcript}"

    yield "🧠 Understanding your emergency...", transcript_md, "", None, ""

    triage, ai_note = groq_triage(transcript, reply_lang)
    if triage:
        service = EMERGENCY_DIRECTORY[CATEGORY_INDEX.get(triage["category"], 0)]
        reply_text = triage["reply"]
        steps = triage["steps"]
        urgency = triage["urgency"]
    else:
        matches = match_emergency(transcript)
        service = matches[0] if matches else EMERGENCY_DIRECTORY[0]
        reply_text = rt["say_call"].format(number=" ".join(service["number"]))
        steps = []
        urgency = ""

    reply_md = f"{rt['audio_response']}\n\n{reply_text}"
    card = result_card(service, steps, urgency, rt)
    yield "🔊 Preparing the spoken reply...", transcript_md, reply_md, None, card

    audio_path, tts_note = timbre_speak(reply_text, reply_lang)
    notes = " ".join(n for n in (ai_note, tts_note) if n)
    status = "✅ Done" + (f" — {notes}" if notes else "")
    yield status, transcript_md, reply_md, audio_path, card


def on_audio(audio_path, lang_code):
    if not audio_path:
        yield "⚠️ No recording found. Please tap the microphone and speak.", "", "", None, ""
        return
    yield "🎧 Listening...", "", "", None, ""
    transcript, err = transcribe(audio_path, lang_code)
    if err:
        yield f"⚠️ {err}", "", "", None, ""
        return
    yield from run_pipeline(transcript, lang_code)


def on_text(text, lang_code):
    text = (text or "").strip()
    if not text:
        yield "⚠️ Please type or speak your emergency first.", "", "", None, ""
        return
    yield from run_pipeline(text, lang_code)


def apply_language(lang_code):
    t = TRANSLATIONS[lang_code]
    return (
        header_html(t),
        gr.update(label=t["mic_header"]),
        gr.update(label=t["type_header"], placeholder=t["type_placeholder"]),
        gr.update(value=t["send"]),
        directory_html(t),
    )


# ---------------------------------------------------------------------------
# Styling (medium, calm colours)
# ---------------------------------------------------------------------------
CSS = """
/* ── Glassmorphism base · Palette: #06141B #11212D #253745 #4A5C6A #9BA8AB #CCD0CF ── */
.gradio-container {
    background: linear-gradient(135deg, #06141B, #11212D, #253745) !important;
    max-width: 1100px !important; margin: auto;
    min-height: 100vh;
}
footer { display: none !important; }

/* Override Gradio default panel/form backgrounds for glass look */
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-block,
.gradio-container .block {
    background: rgba(37,55,69,0.45) !important;
    backdrop-filter: blur(14px) !important; -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(155,168,171,0.15) !important;
    border-radius: 16px !important;
}
.gradio-container label, .gradio-container .label-wrap,
.gradio-container span, .gradio-container p,
.gradio-container .prose, .gradio-container .markdown {
    color: #CCD0CF !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: rgba(17,33,45,0.6) !important;
    border: 1px solid rgba(155,168,171,0.2) !important;
    color: #CCD0CF !important;
    border-radius: 12px !important;
}
.gradio-container button.primary {
    background: linear-gradient(135deg, #4A5C6A, #253745) !important;
    border: 1px solid rgba(155,168,171,0.25) !important; color: #CCD0CF !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(6,20,27,0.5) !important;
}
.gradio-container button.primary:hover {
    background: linear-gradient(135deg, #9BA8AB, #4A5C6A) !important;
    color: #06141B !important;
}

/* ── Header banner (glass) ── */
.header-banner {
    background: rgba(17,33,45,0.55);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(155,168,171,0.2);
    padding: 1.6rem; border-radius: 20px; text-align: center; color: #CCD0CF;
    margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(6,20,27,0.5);
}
.header-banner h1 { color: #CCD0CF !important; margin: 0.2rem 0; text-shadow: 0 2px 10px rgba(6,20,27,0.4); }
.header-banner p { color: #9BA8AB !important; margin: 0.2rem 0; }

/* ── Directory ── */
.dir-title { color: #CCD0CF; margin: 1.2rem 0 0.6rem 0; }
.dir-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; }
.grid-card {
    background: rgba(37,55,69,0.4);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(155,168,171,0.15);
    border-top: 4px solid rgba(155,168,171,0.35);
    border-radius: 16px; padding: 1.1rem; display: flex; flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(6,20,27,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.grid-card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(6,20,27,0.5); }
.grid-card h2 { font-size: 2.4rem; margin: 0 0 8px 0; }
.grid-card h3 { color: #CCD0CF; font-size: 1.05rem; margin: 0 0 8px 0; }
.grid-card p { color: #9BA8AB; font-size: 0.85em; margin: 0 0 12px 0; }
.helpline { color: #9BA8AB; font-weight: bold; margin-bottom: 10px; }

/* ── Result card (glass) ── */
.result-card {
    background: rgba(37,55,69,0.4);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(155,168,171,0.15);
    border-left: 6px solid rgba(155,168,171,0.4);
    border-radius: 16px; padding: 1.2rem;
    box-shadow: 0 4px 20px rgba(6,20,27,0.3);
    color: #CCD0CF;
}
.result-card h3 { margin-top: 0; color: #CCD0CF; }
.result-card p { color: #9BA8AB; }

/* ── Step box ── */
.step-box {
    background: rgba(17,33,45,0.5);
    border: 1px solid rgba(155,168,171,0.12); border-radius: 12px;
    padding: 0.8rem 1rem; margin: 0.8rem 0; color: #9BA8AB; font-size: 0.95em;
}
.step-box ol { margin: 0.4rem 0 0 1.1rem; padding: 0; }

/* ── Call button (glass accent) ── */
a.call-btn {
    display: block; text-align: center;
    background: linear-gradient(135deg, #4A5C6A, #253745);
    color: #CCD0CF !important; text-decoration: none !important;
    padding: 12px 16px; border-radius: 12px; font-weight: bold;
    border: 1px solid rgba(155,168,171,0.2);
    box-shadow: 0 4px 15px rgba(6,20,27,0.4);
    transition: box-shadow 0.2s, transform 0.2s;
}
a.call-btn:hover { transform: translateY(-2px); background: linear-gradient(135deg, #9BA8AB, #4A5C6A); color: #06141B !important; box-shadow: 0 6px 20px rgba(6,20,27,0.5); }

/* ── Urgency pills ── */
.pill {
    display: inline-block; font-size: 0.65em; font-weight: bold; padding: 3px 10px;
    border-radius: 999px; margin-left: 10px; vertical-align: middle; color: #CCD0CF;
    backdrop-filter: blur(8px);
}
.pill-critical { background: rgba(200,16,46,0.7); }
.pill-urgent { background: rgba(204,160,46,0.6); }
.pill-routine { background: rgba(42,157,143,0.6); }

/* ── SOS Panic Button (glass + glow) ── */
.sos-container { text-align: center; margin: 0.8rem 0; }
.sos-btn {
    display: inline-block;
    background: rgba(200,16,46,0.6);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    color: #CCD0CF !important;
    font-size: 1.3rem; font-weight: bold; padding: 18px 40px;
    border: 2px solid rgba(204,208,207,0.25); border-radius: 50px;
    box-shadow: 0 0 30px rgba(200,16,46,0.4), inset 0 0 20px rgba(200,16,46,0.15);
    cursor: pointer; text-decoration: none !important;
    animation: sos-pulse 1.5s ease-in-out infinite;
    letter-spacing: 2px;
    transition: transform 0.2s;
}
.sos-btn:hover { transform: scale(1.05); background: rgba(160,13,36,0.7); }
@keyframes sos-pulse {
    0%, 100% { box-shadow: 0 0 30px rgba(200,16,46,0.4), inset 0 0 20px rgba(200,16,46,0.15); }
    50% { box-shadow: 0 0 50px rgba(200,16,46,0.7), 0 0 80px rgba(200,16,46,0.25), inset 0 0 20px rgba(200,16,46,0.2); }
}
.sos-sub { color: #9BA8AB; font-size: 0.85em; margin-top: 6px; }
"""

FORCE_LIGHT_JS = """() => {
    document.body.classList.remove('dark');

    // ---------- SOS Siren via Web Audio API ----------
    window._sosCtx = null;
    window._sosOsc = null;
    window._sosPlaying = false;

    window.startSiren = function() {
        if (window._sosPlaying) { window.stopSiren(); return; }
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const gain = ctx.createGain();
        gain.gain.value = 1.0;
        gain.connect(ctx.destination);

        const osc = ctx.createOscillator();
        osc.type = 'sawtooth';
        osc.frequency.value = 800;
        osc.connect(gain);
        osc.start();

        // Siren sweep: oscillate between 800 Hz and 1600 Hz
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

        // Auto-stop after 15 seconds
        setTimeout(function() { window.stopSiren(); }, 15000);
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
# UI
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(
    title="Sahaya Emergency Assistant",
    theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone"),
    css=CSS,
    js=FORCE_LIGHT_JS,
) as demo:
    header = gr.HTML(header_html(t0))

    gr.HTML(
        '<div class="sos-container">'
        '<button class="sos-btn" onclick="startSiren(); setTimeout(()=>window.open(\'tel:1091\'),500);">'
        '🚨 SOS PANIC — TAP FOR LOUD ALARM + CALL 1091'
        '</button>'
        '<p class="sos-sub">Triggers a loud siren and dials Assault Helpline 1091</p>'
        '</div>'
    )

    lang_dd = gr.Dropdown(
        choices=[(label, code) for code, label in LANGUAGES.items()],
        value="en_IN",
        label="🌐 Language / ಭಾಷೆ / भाषा",
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

    status = gr.Markdown("")
    transcript_out = gr.Markdown("")
    reply_out = gr.Markdown("")
    reply_audio = gr.Audio(
        label="🔊",
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

# One request at a time keeps us inside Gnani's per-account rate limit.
demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
