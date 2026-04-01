import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "anime_character_classifier.h5"
IMG_SIZE = (300, 300)

CLASS_NAMES = [
    "gojo", "goku", "hinata", "ichigo", "levi",
    "luffy", "naruto", "sasuke", "tanjiro", "zoro"
]

CHARACTER_INFO = {
    "gojo": {
        "name": "Gojo Satoru",
        "anime": "Jujutsu Kaisen",
        "trivia": "Satoru Gojo is a fictional character from Gege Akutami's manga and anime series Jujutsu Kaisen .Gojo is the strongest sorcerer in Jujutsu Kaisen and possesses the Six Eyes and Limitless technique.Gojo takes the role of mentor for the student Yuji Itadori who suffers a curse of Sukuna, helping him become stronger while protecting other characters in the series.",
        "link": "https://www.crunchyroll.com/series/GRDV0019R/jujutsu-kaisen"
    },
    "goku": {
        "name": "Son Goku",
        "anime": "Dragon Ball",
        "trivia": "Son Goku is a fictional character and the main protagonist of the Dragon Ball manga series created by Akira Toriyama. He is based on Sun Wukong, a main character of the classic 16th-century Chinese novel Journey to the West, combined with influences from the Hong Kong action cinema of Jackie Chan and Bruce Lee.Goku is a Saiyan raised on Earth who constantly seeks stronger opponents and self-improvement.",
        "link": "https://www.crunchyroll.com/series/GR19V7816/dragon-ball-super"
    },
    "hinata": {
        "name": "Hinata Shōyō",
        "anime": "Haikyuu!!",
        "trivia": "Shoyo Hinata is a fictional character and the main protagonist of the manga series Haikyu!! created by Haruichi Furudate. Shoyo is a high school student who wishes to become like the Little Giant, a former Karasuno High School student and volleyball club member. To achieve his dream, he decides to join Karasuno, but to join the volleyball team he and Tobio Kageyama, a previous volleyball match opponent, must overcome their rivalry and work together.",
        "link": "https://www.crunchyroll.com/series/GY8VM8MWY/haikyu"
    },
    "ichigo": {
        "name": "Ichigo Kurosaki",
        "anime": "Bleach",
        "trivia": "Ichigo Kurosaki is a fictional character and the main protagonist of the Bleach manga series and its adaptations created by author Tite Kubo. He is a teenage boy with ginger hair who receives Soul Reaper powers after meeting Rukia Kuchiki, a Soul Reaper assigned to patrol around the fictional city of Karakura Town.Ichigo gains Soul Reaper powers and protects both the living and spirit worlds.",
        "link": "https://www.crunchyroll.com/series/G63VGG2NY/bleach"
    },
    "levi": {
        "name": "Levi Ackerman",
        "anime": "Attack on Titan",
        "trivia": "Levi Ackerman is a fictional character from Hajime Isayama's manga series Attack on Titan. Levi is a soldier working for the Survey Corps Special Operations Squad, also known as Squad Levi, a squad of four elite soldiers with impressive combat records hand-picked by him. Humanity’s strongest soldier, Levi is unmatched in combat against Titans.",
        "link": "https://www.crunchyroll.com/series/GR751KNZY/attack-on-titan"
    },
    "luffy": {
        "name": "Monkey D. Luffy",
        "anime": "One Piece",
        "trivia": "Monkey D. Luffy, also known as Straw Hat Luffy, is a fictional character and the protagonist of the Japanese manga series One Piece, created by Eiichiro Oda, as well as the central character of the franchise generated from it.Luffy dreams of becoming the Pirate King and has rubber-like powers from the Gum-Gum Fruit.",
        "link": "https://www.crunchyroll.com/series/GRMG8ZQZR/one-piece"
    },
    "naruto": {
        "name": "Naruto Uzumaki",
        "anime": "Naruto",
        "trivia": "Naruto Uzumaki is the titular protagonist of the manga series Naruto, created by Masashi Kishimoto. He is a ninja from the fictional Hidden Leaf Village.Naruto dreams of becoming Hokage and gaining recognition from his village.",
        "link": "https://www.crunchyroll.com/series/GY9PJ5KWR/naruto"
    },
    "sasuke": {
        "name": "Sasuke Uchiha",
        "anime": "Naruto",
        "trivia": "Sasuke Uchiha is a fictional character in the Naruto manga and anime franchise created by Masashi Kishimoto. Sasuke belongs to the Uchiha clan, a notorious ninja family, and one of the most powerful, allied with Konohagakure.Sasuke is driven by revenge and later redemption, possessing powerful Sharingan abilities.",
        "link": "https://www.crunchyroll.com/series/GY9PJ5KWR/naruto"
    },
    "tanjiro": {
        "name": "Tanjiro Kamado",
        "anime": "Demon Slayer",
        "trivia": "Tanjiro Kamado is a fictional character and the main protagonist of the manga series Demon Slayer: Kimetsu no Yaiba, created by Koyoharu Gotouge. Tanjiro goes on a quest to restore the humanity of his sister, Nezuko, after his family was killed and his sister was transformed into a demon by Muzan Kibutsuji following an attack that resulted in the death of his other relatives.",
        "link": "https://www.crunchyroll.com/series/GY5P48XEY/demon-slayer-kimetsu-no-yaiba"
    },
    "zoro": {
        "name": "Roronoa Zoro",
        "anime": "One Piece",
        "trivia": "Roronoa Zoro, also known as Pirate Hunter Zoro, is a fictional character in the manga series and media franchise One Piece created by Eiichiro Oda.Zoro aims to become the world’s greatest swordsman and fights using three swords.",
        "link": "https://www.crunchyroll.com/series/GRMG8ZQZR/one-piece"
    }
}

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Anime Character AI", page_icon="🎌", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f172a, #020617);
    color: white;
}
.main {
    padding-top: 2rem;
}
.title-text {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle-text {
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 2rem;
    color: #cbd5f5;
}
.upload-box {
    border: 2px dashed #64748b;
    padding: 2rem;
    border-radius: 16px;
    background-color: #020617;
}
.result-card {
    background: linear-gradient(135deg, #020617, #020617);
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 0 40px rgba(56,189,248,0.25);
}
.badge {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    background: #38bdf8;
    color: black;
    font-weight: 600;
    font-size: 0.9rem;
}
.confidence {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f472b6;
}
.watch-btn a {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.6rem 1.2rem;
    background: linear-gradient(90deg, #38bdf8, #f472b6);
    border-radius: 999px;
    color: black !important;
    text-decoration: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="title-text">🎌 Cartoon Character AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload an Cartoon face and discover the character, trivia, and where to watch.</div>', unsafe_allow_html=True)

# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1, 1.2])

with left:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload an anime image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with right:
    if uploaded_file:
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        idx = np.argmax(preds)
        confidence = preds[idx] * 100
        key = CLASS_NAMES[idx]
        info = CHARACTER_INFO[key]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<span class='badge'>Prediction</span>", unsafe_allow_html=True)
        st.markdown(f"## {info['name']}")
        st.markdown(f"**Anime:** {info['anime']}")
        st.markdown(f"<div class='confidence'>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

        st.markdown("### ✨ Trivia")
        st.write(info["trivia"])

        st.markdown(f"<div class='watch-btn'><a href='{info['link']}' target='_blank'>▶ Watch {info['anime']}</a></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("⚡ Powered by EfficientNetB3 • Built by You")