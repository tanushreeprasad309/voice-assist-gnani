"""
Sahaya Emergency Assistant (Gradio on Render)
Voice-first emergency triage and first aid for rural India.

Gnani AI stack:
  Prisma  (speech-to-text)   -> https://api.vachana.ai/stt/v3
  Timbre  (text-to-speech)   -> https://api.vachana.ai/api/v1/tts/inference
Groq (free triage reasoning) -> https://api.groq.com/openai/v1/chat/completions

Environment variables to set in Render Dashboard -> Environment:
  GNANI_API_KEY   Prisma + Timbre key
  GROQ_API_KEY    Free key from console.groq.com
Optional:
  GROQ_MODEL      default llama-3.3-70b-versatile (falls back automatically)
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

# Groq settings (free tier) with automatic model fallback
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

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
        "type_placeholder": "e.g. Need an ambulance, snake bite, or accident",
        "send": "Send Emergency Alert",
        "detected": "🗣️ **Detected Emergency:**",
        "action_header": "⚡ Recommended Action & First-Aid",
        "first_aid_label": "🩹 Immediate First-Aid Steps:",
        "call_btn": "📞 Call {number} Now",
        "directory_header": "📋 Emergency Directory (Tap Any Card to Call)",
        "helpline": "Helpline:",
        "audio_response": "🔊 **Sahaya's Response:**",
        "say_call": "Please stay calm and call {number} immediately.",
    },
    "kn_IN": {
        "title": "🚨 ಸಹಾಯ ತುರ್ತು ಸಹಾಯಕ",
        "subtitle": "ಬಹುಭಾಷಾ ಧ್ವನಿ ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ ಮತ್ತು ತುರ್ತು ನೆರವು",
        "mic_header": "🎙️ ನಿಮ್ಮ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯನ್ನು ಹೇಳಿ",
        "mic_start": "🔴 ಮಾತನಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "mic_stop": "⏹️ ನಿಲ್ಲಿಸಿ",
        "type_header": "⌨️ ಅಥವಾ ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ",
        "type_placeholder": "ಉದಾ. ನನಗೆ ಹಾವು ಕಚ್ಚಿದೆ, ಆಂಬ್ಯುಲೆನ್ಸ್ ಬೇಕು",
        "send": "ಕಳುಹಿಸಿ",
        "detected": "🗣️ **ಪತ್ತೆಯಾದ ತುರ್ತು ಪರಿಸ್ಥಿತಿ:**",
        "action_header": "⚡ ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ ಮತ್ತು ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ",
        "first_aid_label": "🩹 ಪ್ರಥಮ ಚಿಕಿತ್ಸಾ ಕ್ರಮಗಳು:",
        "call_btn": "📞 ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ",
        "directory_header": "📋 ತುರ್ತು ಸಹಾಯವಾಣಿಗಳು (ಕರೆ ಮಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ)",
        "helpline": "ಸಹಾಯವಾಣಿ:",
        "audio_response": "🔊 **ಸಹಾಯ ಪ್ರತಿಕ್ರಿಯೆ:**",
        "say_call": "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ, ತಕ್ಷಣ {number} ಗೆ ಕರೆ ಮಾಡಿ.",
    },
    "hi_IN": {
        "title": "🚨 सहायता आपातकालीन सहायक",
        "subtitle": "बहुभाषी वॉयस फर्स्ट-एड और आपातकालीन प्रेषण",
        "mic_header": "🎙️ अपनी आपात स्थिति बोलें",
        "mic_start": "🔴 बोलने के लिए टैप करें",
        "mic_stop": "⏹️ रोकें",
        "type_header": "⌨️ या टाइप करें",
        "type_placeholder": "जैसे एम्बुलेंस चाहिए, सांप ने काटा है, दुर्घटना हुई है",
        "send": "भेजें",
        "detected": "🗣️ **पता चली आपात स्थिति:**",
        "action_header": "⚡ अनुशंसित कार्रवाई और प्राथमिक उपचार",
        "first_aid_label": "🩹 प्राथमिक उपचार दिशानिर्देश:",
        "call_btn": "📞 अभी {number} पर कॉल करें",
        "directory_header": "📋 आपातकालीन डायरेक्टरी (कॉल करने के लिए टैप करें)",
        "helpline": "हेल्पलाइन:",
        "audio_response": "🔊 **सहायता की प्रतिक्रिया:**",
        "say_call": "कृपया शांत रहें और तुरंत {number} पर कॉल करें।",
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
        "audio_response": "🔊 **சஹாயாவின் பதில்:**",
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
        "first_aid_label": "🩹 ಪ್ರಥమ చికిత్స మార్గదర్శకాలు:",
        "call_btn": "📞 ఇప్పుడే {number} కి కాల్ చేయండి",
        "directory_header": "📋 అత్యవసర డైరెక్టరీ (కాల్ చేయడానికి ట్యాప్ చేయండి)",
        "helpline": "హెల్ప్‌లైన్:",
        "audio_response": "🔊 **సహాయ ప్రతిస్పందన:**",
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
        "audio_response": "🔊 **സഹായയുടെ പ്രതികരണം:**",
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
        "audio_response": "🔊 **साहाय्यचा प्रतिसाद:**",
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
        "audio_response": "🔊 **સહાયનો જવાબ:**",
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
        "audio_response": "🔊 **সহায়ের প্রতিক্রিয়া:**",
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
        "helpline": "ਹੈਲਪਲਾਈನ್:",
        "audio_response": "🔊 **ਸਹਾਇਤਾ ਦਾ ਜਵਾਬ:**",
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
        "audio_response": "🔊 **ସହାୟଙ୍କ ପ୍ରତିକ୍ରିୟା:**",
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
}

TIMBRE_LANGS = {
    "en_IN", "hi_IN", "ta_IN", "te_IN", "kn_IN",
    "ml_IN", "mr_IN", "pa_IN", "bn_IN", "gu_IN",
}

# ---------------------------------------------------------------------------
# Localized Offline Fallback Responses (Ensures Native Language Output)
# ---------------------------------------------------------------------------
LOCALIZED_FALLBACK_STEPS = {
    "kn_IN": {
        "snake": (
            "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ. ತಕ್ಷಣ ಒಂದು ಸೊನ್ನೆ ಎಂಟು ಆಂಬ್ಯುಲೆನ್ಸ್‌ಗೆ ಕರೆ ಮಾಡಿ. ಕಚ್ಚಿದ ಭಾಗವನ್ನು ಅಲುಗಾಡಿಸಬೇಡಿ.",
            [
                "ವ್ಯಕ್ತಿಯನ್ನು ಶಾಂತವಾಗಿ ಮಲಗಿಸಿ ಮತ್ತು ಕಚ್ಚಿದ ಭಾಗವನ್ನು ಹೃದಯದ ಮಟ್ಟಕ್ಕಿಂತ ಕೆಳಗಿರಿಸಿ.",
                "ಗಾಯದ ಮೇಲೆ ಬಟ್ಟೆಯನ್ನು ಬಿಗಿಯಾಗಿ ಕಟ್ಟಬೇಡಿ ಅಥವಾ ಬಾಯಿಯಿಂದ ವಿಷ ಹೀರಬೇಡಿ.",
                "ಉಂಗುರ, ಬಳೆ ಅಥವಾ ಬಿಗಿಯಾದ ಬಟ್ಟೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ತಕ್ಷಣ ಆಸ್ಪತ್ರೆಗೆ ಕರೆದೊಯ್ಯಿರಿ.",
            ],
        ),
        "default": (
            "ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ. ತಕ್ಷಣ ಸಹಾಯವಾಣಿಗೆ ಕರೆ ಮಾಡಿ ಮತ್ತು ಕೆಳಗಿನ ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ ಅನುಸರಿಸಿ.",
            [
                "ವ್ಯಕ್ತಿಯ ಉಸಿರಾಟ ಮತ್ತು ಪ್ರಜ್ಞೆಯನ್ನು ಪರಿಶೀಲಿಸಿ, ಅವರನ್ನು ಶಾಂತವಾಗಿರಿಸಿ.",
                "ರಕ್ತಸ್ರಾವವಾಗುತ್ತಿದ್ದರೆ ಸ್ವಚ್ಛವಾದ ಬಟ್ಟೆಯಿಂದ ಗಾಯದ ಮೇಲೆ ಒತ್ತಿ ಹಿಡಿಯಿರಿ.",
                "ಆಂಬ್ಯುಲೆನ್ಸ್ ಬರುವವರೆಗೆ ವ್ಯಕ್ತಿಯನ್ನು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ಇರಿಸಿ.",
            ],
        ),
    },
    "hi_IN": {
        "snake": (
            "कृपया घबराएं नहीं। तुरंत १०८ एम्बुलेंस को कॉल करें। सांप के काटे 
 हुए हिस्से को हिलाएं नहीं।",
            [
                "व्यक्ति को शांत लिटाएं और कटे हुए हिस्से को दिल के स्तर से नीचे रखें।",
                "घाव पर कपड़ा कसकर न बांधें और मुंह से जहर न चूसें।",
                "अंगूठी, कंगन या तंग कपड़े उतार दें और तुरंत अस्पताल ले जाएं।",
            ],
        ),
        "default": (
            "कृपया घबराएं नहीं। तुरंत आपातकालीन हेल्पलाइन पर कॉल करें और प्राथमिक उपचार करें।",
            [
                "व्यक्ति की सांस और होश की जांच करें और उन्हें शांत रखें।",
                "यदि खून बह रहा है, तो साफ कपड़े से घाव पर दबाव बनाएं।",
                "मदद आने तक व्यक्ति को सुरक्षित स्थान पर रखें।",
            ],
        ),
    },
    "ta_IN": {
        "default": (
            "தயவுசெய்து பதற்றப்பட வேண்டாம். உடனடியாக 108 ஐ அழைத்து முதலுதவி செய்யுங்கள்.",
            [
                "நபரின் சுவாசத்தை சரிபார்த்து அவரை அமைதியாக வைத்திருங்கள்.",
                "இரத்தக்கசிவு இருந்தால் சுத்தமான துணியால் அழுத்திப் பிடிக்கவும்.",
                "உதவி வரும் வரை பாதுகாப்பான இடத்தில் இருக்கவும்.",
            ],
        ),
    },
    "te_IN": {
        "default": (
            "దయచేసి కంగారు పడకండి. వెంటనే 108 కి కాల్ చేసి ప్రథమ చికిత్స అందించండి.",
            [
                "వ్యక్తి శ్వాసను పరిశీలించి వారిని ప్రశాంతంగా ఉంచండి.",
                "రక్తస్రావం అవుతుంటే శుభ్రమైన గుడ్డతో గట్టిగా నొక్కి ఉంచండి.",
                "అంబులెన్స్ వచ్చేవరకు సురక్షిత ప్రదేశంలో ఉంచండి.",
            ],
        ),
    },
    "ml_IN": {
        "default": (
            "ദയവായി പരിഭ്രാന്തരാകരുത്. ഉടൻ തന്നെ 108-ൽ വിളിച്ച് പ്രഥമശുശ്രൂഷ നൽകുക.",
            [
                "വ്യക്തിയുടെ ശ്വാസോച്ഛ്വാസം പരിശോധിച്ച് സമാധാനിപ്പിക്കുക.",
                "രക്തസ്രാവമുണ്ടെങ്കിൽ വൃത്തിയുള്ള തുണി ഉപയോഗിച്ച് അമർത്തിപ്പിടിക്കുക.",
                "സഹായം എത്തുന്നതുവരെ സുരക്ഷിത സ്ഥാനത്ത് തുടരുക.",
            ],
        ),
    },
    "mr_IN": {
        "default": (
            "कृपया घाबरू नका. ताबडतोब १०८ वर कॉल करा आणि प्रथमोपचार सुरू करा.",
            [
                "व्यक्तीचा श्वासोच्छवास तपासा आणि त्यांना शांत ठेवा.",
                "रक्तस्त्राव होत असल्यास स्वच्छ कापडाने जखमेवर दाब द्या.",
                "मदत येईपर्यंत सुरक्षित ठिकाणी थांबा.",
            ],
        ),
    },
    "gu_IN": {
        "default": (
            "કૃપા કરીને ગભરાશો નહીં. તરત જ 108 પર કૉલ કરો અને પ્રાથમિક સારવાર આપો.",
            [
                "વ્યક્તિના શ્વાસની તપાસ કરો અને તેમને શાંત રાખો.",
                "જો લોહી નીકળતું હોય તો સ્વચ્છ કપડાથી ઘા પર દબાણ આપો.",
                "મદદ આવે ત્યાં સુધી સુરક્ષિત જગ્યાએ રહો.",
            ],
        ),
    },
    "bn_IN": {
        "default": (
            "দয়া করে আতঙ্কিত হবেন না। অবিলম্বে ১০৮-এ কল করুন এবং প্রাথমিক চিকিৎসা দিন।",
            [
                "ব্যক্তির শ্বাস-প্রশ্বাস পরীক্ষা করুন এবং তাকে শান্ত রাখুন।",
                "রক্তপাত হলে পরিষ্কার কাপড় দিয়ে ক্ষতস্থানে চাপ দিয়ে ধরুন।",
                "সাহায্য না আসা পর্যন্ত নিরাপদ স্থানে থাকুন।",
            ],
        ),
    },
    "pa_IN": {
        "default": (
            "ਕਿਰਪਾ ਕਰਕੇ ਘਬਰਾਓ ਨਾ। ਤੁਰੰਤ 108 'ਤੇ ਕਾਲ ਕਰੋ ਅਤੇ ਮੁੱਢਲੀ ਸਹਾਇਤਾ ਦਿਓ।",
            [
                "ਵਿਅਕਤੀ ਦੇ ਸਾਹ ਦੀ ਜਾਂਚ ਕਰੋ ਅਤੇ ਉਨ੍ਹਾਂ ਨੂੰ ਸ਼ਾਂਤ ਰੱਖੋ।",
                "ਜੇ ਖੂਨ ਵਹਿ ਰਿਹਾ ਹੈ ਤਾਂ ਸਾਫ਼ ਕੱਪੜੇ ਨਾਲ ਜ਼ਖ਼ਮ 'ਤੇ ਦਬਾਅ ਪਾਓ।",
                "ਮਦਦ ਆਉਣ ਤੱਕ ਸੁਰੱਖਿਅਤ ਥਾਂ 'ਤੇ ਰਹੋ।",
            ],
        ),
    },
    "or_IN": {
        "default": (
            "ଦୟାକରି ଭୟଭୀତ ହୁଅନ୍ତୁ ନାହିଁ। ତୁରନ୍ତ 108 କୁ କଲ୍ କରନ୍ତୁ ଏବଂ ପ୍ରଥମ ଚିକିତ୍ସା କରନ୍ତୁ।",
            [
                "ବ୍ୟକ୍ତିଙ୍କ ନିଶ୍ୱାସ ପ୍ରଶ୍ୱାସ ଯାଞ୍ଚ କରନ୍ତୁ ଏବଂ ତାଙ୍କୁ ଶାନ୍ତ ରଖନ୍ତୁ।",
                "ରକ୍ତସ୍ରାବ ହେଉଥିଲେ ସଫା କପଡ଼ାରେ କ୍ଷତ ସ୍ଥାନକୁ ଦାବି ଧରନ୍ତୁ।",
                "ସାହାଯ୍ୟ ଆସିବା ପର୍ଯ୍ୟନ୍ତ ସୁରକ୍ଷିତ ସ୍ଥାନରେ ରୁହନ୍ତୁ।",
            ],
        ),
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
        "keywords": [
            "ambulance", "hospital", "doctor", "accident", "heart", "injury", "medical",
            "bleeding", "unconscious", "faint", "snake", "bite", "poison", "breathing", "pregnan",
            "ಆಂಬ್ಯುಲೆನ್ಸ್", "ಆಸ್ಪತ್ರೆ", "ಅಪಘಾತ", "ಹಾವು", "ಕಚ್ಚಿದೆ", "ರಕ್ತ", "ನೋವು", "ಜ್ವರ",
            "एम्बुलेंस", "अस्पताल", "दुर्घटना", "चोट", "एक्सीडेंट", "सांप", "खून",
            "ആംബുലൻസ്", "പ്പാമ്പ്", "அவசர ஊர்தி", "ஆம்புலன்ஸ்", "பாம்பு", "ఆంబులెన్స్", "పాము",
        ],
        "desc": "Immediate dispatch for medical crises, snake bites, severe injuries, or emergency transport.",
        "first_aid": "1. Check airway, breathing & pulse.\n2. Keep the person still and calm.\n3. Apply firm pressure to bleeding wounds until 108 arrives.",
    },
    {
        "title": "Police & Safety",
        "icon": "👮",
        "number": "112",
        "keywords": [
            "police", "attack", "crime", "theft", "danger", "fight", "robbery",
            "ಪೊಲೀಸ್", "ಕಳ್ಳ", "ಗಲಾಟೆ", "पुलिस", "बचाओ", "പോലീസ്", "காவல்துறை", "పోలీసు", "ପୋଲିସ",
        ],
        "desc": "Immediate assistance for crime, personal safety threats, or law enforcement.",
        "first_aid": "1. Move to a safe, well-lit public area.\n2. Stay on the line with 112 and share nearby landmarks.",
    },
    {
        "title": "Assault & Distress Helpline",
        "icon": "🛡️",
        "number": "1091",
        "keywords": [
            "assault", "harassment", "abuse", "stalking", "threat", "molestation",
            "ದೌರ್ಜನ್ಯ", "ಕಿರುಕುಳ", "उत्पीड़न", "छेड़छाड़", "ഉപദ്രവം", "துன்புறுத்தல்",
            "వేధింపులు", "ଅତ୍ୟାଚାର",
        ],
        "desc": "Dedicated protection and rapid response for assault, harassment, or severe distress.",
        "first_aid": "1. Seek immediate shelter in a safe shop or public group.\n2. Call 1091 or 112 instantly.\n3. Avoid confronting the person alone.",
    },
    {
        "title": "Fire & Rescue",
        "icon": "🚒",
        "number": "101",
        "keywords": [
            "fire", "burn", "smoke", "cylinder", "blast", "rescue", "gas",
            "ಬೆಂಕಿ", "ಸುಟ್ಟ", "आग", "തീ", "தீ", "అగ్ని", "ନିଆଁ",
        ],
        "desc": "Emergency response for fires, gas leaks, burns, or structural entrapment.",
        "first_aid": "1. Evacuate immediately and stay low under smoke.\n2. Run cool water over minor burns for 10-15 minutes.",
    },
    {
        "title": "Disaster & Flood Relief",
        "icon": "🌊",
        "number": "1078",
        "keywords": [
            "flood", "earthquake", "cyclone", "landslide", "disaster",
            "ಭೂಕಂಪ", "ಪ್ರವಾಹ", "बाढ़", "പ്രളയം", "வெள்ளம்", "వరద", "ବନ୍ୟା",
        ],
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
    (0x0900, 0x097F, "hi_IN"),
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
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return fallback
    best = max(counts, key=counts.get)
    if best == "hi_IN" and fallback == "mr_IN":
        return "mr_IN"
    return best


def match_emergency(text):
    if not text:
        return []
    lower = text.lower()
    return [s for s in EMERGENCY_DIRECTORY if any(kw in lower for kw in s["keywords"])]


def get_fallback_response(transcript, lang_code, service):
    lang_pack = LOCALIZED_FALLBACK_STEPS.get(lang_code)
    if lang_pack:
        if "ಹಾವು" in transcript or "snake" in transcript.lower() or "सांप" in transcript.lower():
            reply, steps = lang_pack.get("snake", lang_pack["default"])
        else:
            reply, steps = lang_pack["default"]
        return reply, steps
    
    # Generic fallback if language pack isn't fully defined
    fallback_texts = {
        "hi_IN": ("कृपया घबराएं नहीं। तुरंत १०८ पर कॉल करें।", ["सांस और पल्स जांचें", "व्यक्ति को शांत रखें", "तुरंत सहायता लें"]),
        "kn_IN": ("ದಯವಿಟ್ಟು ಗಾಬರಿಯಾಗಬೇಡಿ. ತಕ್ಷಣ 108 ಗೆ ಕರೆ ಮಾಡಿ.", ["ಉಸಿರಾಟವನ್ನು ಪರಿಶೀಲಿಸಿ", "ವ್ಯಕ್ತಿಯನ್ನು ಶಾಂತವಾಗಿರಿಸಿ", "ತಕ್ಷಣ ಸಹಾಯ ಪಡೆಯಿರಿ"]),
        "ta_IN": ("தயவுசெய்து பதற்றப்பட வேண்டாம். உடனடியாக 108 ஐ அழைக்கவும்.", ["சுவாசத்தை சரிபார்க்கவும்", "அமைதியாக இருக்கவும்", "உதவி பெறவும்"]),
        "te_IN": ("దయచేసి కంగారు పడకండి. వెంటనే 108 కి కాల్ చేయండి.", ["శ్వాసను పరిశీలించండి", "ప్రశాంతంగా ఉంచండి", "సహాయం పొందండి"]),
        "en_IN": ("Please stay calm and call 108 immediately.", ["Check breathing & pulse", "Keep person calm", "Seek immediate help"])
    }
    return fallback_texts.get(lang_code, fallback_texts["en_IN"])


# ---------------------------------------------------------------------------
# Gnani Prisma: Speech to Text
# ---------------------------------------------------------------------------
def transcribe(audio_path, lang_code):
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return "", "GNANI_API_KEY is missing."
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
        return "", f"Could not reach speech service: {e}"
    if r.status_code != 200:
        return "", f"Speech recognition failed ({r.status_code})"
    try:
        data = r.json()
    except ValueError:
        return "", "Unexpected response from speech service."
    transcript = str(data.get("transcript") or data.get("text") or "").strip()
    if not transcript:
        return "", "No speech detected. Please try again."
    return transcript, ""


# ---------------------------------------------------------------------------
# Groq Triage (Strict Native Language Enforcement)
# ---------------------------------------------------------------------------
def groq_triage(transcript, lang_code):
    groq_key = get_env("GROQ_API_KEY")
    if not groq_key:
        return None, "Groq key missing"

    language_name = LANG_NAMES.get(lang_code, "English")
    system_prompt = (
        f"You are an empathetic emergency assistant for rural India. "
        f"CRITICAL INSTRUCTION: You MUST reply entirely in {language_name} using its native script (e.g., Devanagari for Hindi/Marathi, Kannada script for Kannada, etc.). DO NOT reply in English or Latin script. "
        "Analyze the user's emergency and output valid JSON ONLY with these exact keys: "
        '{"category": "medical|police|assault|fire|disaster", '
        '"urgency": "critical|urgent|routine", '
        '"reply": "short comforting instructions strictly in the target native script", '
        '"steps": ["step 1 in native script", "step 2 in native script", "step 3 in native script"]}'
    )

    custom_model = get_env("GROQ_MODEL")
    models_to_try = ([custom_model] if custom_model else []) + [
        m for m in GROQ_MODELS if m != custom_model
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_key}",
    }

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Emergency: {transcript}"},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }
        try:
            r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=25)
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
            raw_steps = data.get("steps", [])
            steps = [str(s).strip() for s in raw_steps if str(s).strip()][:5]
            return {
                "category": str(data.get("category", "medical")).strip().lower(),
                "urgency": str(data.get("urgency", "urgent")).strip().lower(),
                "reply": reply,
                "steps": steps,
            }, ""
        except Exception:
            continue

    return None, "Offline fallback active"


# ---------------------------------------------------------------------------
# Gnani Timbre: Text to Speech
# ---------------------------------------------------------------------------
def timbre_speak(text, lang_code):
    if lang_code not in TIMBRE_LANGS:
        return None, "Voice not available for this language"
    api_key = get_env("GNANI_API_KEY")
    if not api_key:
        return None, "API key missing"

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
    except Exception:
        return None, "TTS connection failed"
    if r.status_code != 200:
        return None, "TTS generation failed"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(r.content)
    return tmp.name, ""


# ---------------------------------------------------------------------------
# HTML Builders
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
        f'<h3>{service["icon"]} {html.escape(rt["action_header"])}{pill}</h3>',
        f'<p>{html.escape(service["title"])} — {html.escape(service["desc"])}</p>',
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
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(transcript, selected_lang):
    reply_lang = detect_lang(transcript, selected_lang)
    rt = TRANSLATIONS.get(reply_lang, TRANSLATIONS["en_IN"])
    transcript_md = f"{rt['detected']} {transcript}"

    yield "", transcript_md, "", None, ""

    triage, _ = groq_triage(transcript, reply_lang)
    if triage:
        service = EMERGENCY_DIRECTORY[CATEGORY_INDEX.get(triage["category"], 0)]
        reply_text = triage["reply"]
        steps = triage["steps"]
        urgency = triage["urgency"]
    else:
        matches = match_emergency(transcript)
        service = matches[0] if matches else EMERGENCY_DIRECTORY[0]
        reply_text, steps = get_fallback_response(transcript, reply_lang, service)
        urgency = "urgent"

    reply_md = f"{rt['audio_response']}\n\n{reply_text}"
    card = result_card(service, steps, urgency, rt)
    yield "", transcript_md, reply_md, None, card

    audio_path, _ = timbre_speak(reply_text, reply_lang)
    yield "", transcript_md, reply_md, audio_path, card


def on_audio(audio_path, lang_code):
    if not audio_path:
        yield "⚠️ No recording found.", "", "", None, ""
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
        yield "⚠️ Please enter your emergency first.", "", "", None, ""
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
# Styling
# ---------------------------------------------------------------------------
CSS = """
body, .gradio-container {
    background: linear-gradient(135deg, #07111E 0%, #0F233C 50%, #16325B 100%) !important;
    color: #FFFFFF !important;
    max-width: 1120px !important;
    margin: auto !important;
    min-height: 100vh;
    font-family: system-ui, -apple-system, sans-serif;
}
footer { display: none !important; }

.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-block,
.gradio-container .block {
    background: #112643 !important;
    border: 1px solid #2D5380 !important;
    border-radius: 16px !important;
    color: #FFFFFF !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container span,
.gradio-container p,
.gradio-container strong,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container .prose,
.gradio-container .markdown {
    color: #FFFFFF !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #0A182B !important;
    border: 1.5px solid #3B6E9C !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
}

.gradio-container button.primary {
    background: linear-gradient(135deg, #FF6500, #E5383B) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
}

.header-banner {
    background: linear-gradient(135deg, #132B4C, #1D4370);
    border: 1px solid #3E72A8;
    padding: 1.6rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 1rem;
}
.header-banner h1 { font-size: 2rem; margin: 0.2rem 0; }
.header-banner p { color: #D4E6F9 !important; font-size: 1.05rem; }

.response-box {
    background: #132C4E !important;
    border: 1.5px solid #3E72A8 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin: 8px 0 !important;
    color: #FFFFFF !important;
}

.result-card {
    background: #132C4E !important;
    border: 1.5px solid #4A86C5 !important;
    border-left: 7px solid #FF6500 !important;
    border-radius: 16px;
    padding: 1.4rem;
    margin-top: 0.8rem;
}

.step-box {
    background: #0A192F !important;
    border: 1px solid #2D5A88 !important;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    line-height: 1.6;
}
.step-box strong { color: #FFB703 !important; }

.dir-title { margin: 1.5rem 0 0.8rem 0; font-size: 1.3rem; }
.dir-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
}
.grid-card {
    background: #132C4E !important;
    border: 1px solid #356394 !important;
    border-top: 4px solid #38BDF8 !important;
    border-radius: 16px;
    padding: 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.helpline { color: #38BDF8 !important; font-weight: 700; margin-bottom: 10px; }

a.call-btn {
    display: block;
    text-align: center;
    background: linear-gradient(135deg, #10B981, #059669);
    color: #FFFFFF !important;
    text-decoration: none !important;
    padding: 12px 16px;
    border-radius: 12px;
    font-weight: 700;
}

.pill {
    display: inline-block;
    font-size: 0.72em;
    font-weight: 800;
    padding: 4px 12px;
    border-radius: 999px;
    margin-left: 10px;
}
.pill-critical { background: #E11D48; }
.pill-urgent { background: #D97706; }
.pill-routine { background: #0D9488; }

.sos-container { text-align: center; margin: 0.9rem 0; }
.sos-btn {
    display: inline-block;
    background: linear-gradient(135deg, #E11D48, #BE123C);
    color: #FFFFFF !important;
    font-size: 1.25rem;
    font-weight: 800;
    padding: 16px 38px;
    border: 2px solid #FDA4AF;
    border-radius: 50px;
    cursor: pointer;
    text-decoration: none !important;
    animation: sos-pulse 1.5s ease-in-out infinite;
}
@keyframes sos-pulse {
    0%, 100% { box-shadow: 0 0 25px rgba(225, 29, 72, 0.5); }
    50% { box-shadow: 0 0 45px rgba(225, 29, 72, 0.9); }
}
.sos-sub { color: #E2E8F0 !important; font-size: 0.9rem; margin-top: 6px; }
"""

INIT_JS = """() => {
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
# UI Layout
# ---------------------------------------------------------------------------
t0 = TRANSLATIONS["en_IN"]

with gr.Blocks(
    title="Sahaya Emergency Assistant",
    theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    css=CSS,
    js=INIT_JS,
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
        label="🌐 Select Language / ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ / भाषा चुनें",
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
    transcript_out = gr.Markdown("", elem_classes=["response-box"])
    reply_out = gr.Markdown("", elem_classes=["response-box"])
    reply_audio = gr.Audio(
        label="🔊 Sahaya Voice Guidance",
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
