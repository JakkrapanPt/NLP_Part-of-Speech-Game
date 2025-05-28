import streamlit as st
import random
import requests
import spacy
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from typing import List, Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

# Try to import NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import pythainlp
    from pythainlp import word_tokenize
    from pythainlp.tag import pos_tag
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    st.warning("PyThaiNLP not installed. Install with: pip install pythainlp")

@dataclass
class WordInfo:
    def __init__(self, word: str, pos: str, index: int):
        self.word = word
        self.pos = pos
        self.pos_name = ""
        self.user_answer = ""
        self.is_correct = False
        self.index = index

class POSGame:
    def __init__(self):
        self.pos_categories = {
            'en': {
                'NOUN': 'Noun (คำนาม)',
                'PRON': 'Pronoun (คำสรรพนาม)', 
                'VERB': 'Verb (คำกริยา)',
                'ADJ': 'Adjective (คำคุณศัพท์)',
                'ADV': 'Adverb (คำกริยาวิเศษณ์)',
                'ADP': 'Preposition (คำบุพบท)',
                'CONJ': 'Conjunction (คำสันธาน)',
                'INTJ': 'Interjection (คำอุทาน)'
            },
            'th': {
                'NOUN': 'คำนาม (Noun)',
                'PRON': 'คำสรรพนาม (Pronoun)',
                'VERB': 'คำกริยา (Verb)', 
                'ADJ': 'คำคุณศัพท์ (Adjective)',
                'ADV': 'คำกริยาวิเศษณ์ (Adverb)',
                'ADP': 'คำบุพบท (Preposition)',
                'CONJ': 'คำสันธาน (Conjunction)',
                'INTJ': 'คำอุทาน (Interjection)'
            }
        }
        
        # Sample sentences for demo (when LLM is not available)
        self.sample_sentences = {
            'en': {
                'easy': [
                    "The cat sleeps peacefully.",
                    "She runs quickly today.",
                    "Birds fly high above.",
                    "We eat delicious food."
                ],
                'medium': [
                    "The beautiful flowers bloom in spring garden.",
                    "Students study hard for their important exams.",
                    "My grandmother tells interesting stories every evening."
                ],
                'hard': [
                    "The magnificent orchestra performed brilliantly at the prestigious concert hall last night.",
                    "Scientists carefully analyze complex data to understand mysterious phenomena in deep space."
                ]
            },
            'th': {
                'easy': [
                    "แมวนอนหลับสบาย",
                    "เขาวิ่งเร็วมาก",
                    "นกบินสูงขึ้น",
                    "เรากินข้าวอร่อย"
                ],
                'medium': [
                    "นักเรียนขยันอ่านหนังสือเพื่อสอบ",
                    "ดอกไม้สวยบานในสวนหลังบ้าน",
                    "คุณยายเล่านิทานให้ฟังทุกคืน"
                ],
                'hard': [
                    "นักวิทยาศาสตร์วิเคราะห์ข้อมูลซับซ้อนเพื่อทำความเข้าใจปรากฏการณ์ลึกลับในอวกาศ",
                    "วงดุริยางค์ชื่อดังแสดงอย่างยอดเยี่ยมในหอประชุมใหญ่เมื่อคืนที่ผ่านมา"
                ]
            }
        }
        
        # Load spaCy models if available
        if SPACY_AVAILABLE:
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp_en = None
        else:
            self.nlp_en = None
    
    def generate_sentence_with_llm(self, language: str, difficulty: str, api_url: str = None) -> tuple:
        """Generate sentence using LLM API or fallback to samples
        
        Returns:
            tuple: (sentence, actual_difficulty)
        """
        # Try to call API if URL is provided
        if api_url:
            try:
                # ส่งเฉพาะ language ไปยัง API ใหม่ (ไม่ต้องส่ง difficulty)
                response = requests.post(
                    f"{api_url}/generate_sentence",
                    json={
                        'language': language
                    },
                    timeout=75
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and data.get('sentence'):
                        # ส่งคืนทั้งประโยคและระดับความยากที่ได้จาก API
                        actual_difficulty = data.get('difficulty', difficulty)
                        return data['sentence'], actual_difficulty
                    else:
                        st.warning(f"API returned error: {data.get('error', 'Unknown error')}")
                else:
                    st.warning(f"API request failed with status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.warning("API request timed out. Using fallback sentences.")
            except requests.exceptions.ConnectionError:
                st.warning("Could not connect to API. Using fallback sentences.")
            except Exception as e:
                st.warning(f"API error: {str(e)}. Using fallback sentences.")
        
        # Fallback to sample sentences
        sentences = self.sample_sentences[language][difficulty]
        return random.choice(sentences), difficulty
    
    def analyze_sentence_english(self, sentence: str) -> List[WordInfo]:
        """Analyze English sentence using spaCy"""
        if not self.nlp_en:
            # Fallback analysis
            words = sentence.replace('.', '').replace(',', '').split()
            return [WordInfo(word=word, pos='NOUN', index=i) for i, word in enumerate(words)]
        
        doc = self.nlp_en(sentence)
        word_infos = []
        
        for i, token in enumerate(doc):
            if token.is_alpha:  # Only process alphabetic tokens
                # Map spaCy POS to our simplified categories
                pos_mapping = {
                    'NOUN': 'NOUN', 'PROPN': 'NOUN',
                    'PRON': 'PRON',
                    'VERB': 'VERB', 'AUX': 'VERB',
                    'ADJ': 'ADJ',
                    'ADV': 'ADV',
                    'ADP': 'ADP',
                    'CONJ': 'CONJ', 'CCONJ': 'CONJ', 'SCONJ': 'CONJ',
                    'INTJ': 'INTJ'
                }
                
                simplified_pos = pos_mapping.get(token.pos_, 'NOUN')
                word_infos.append(WordInfo(
                    word=token.text,
                    pos=simplified_pos,
                    index=i
                ))
        
        return word_infos
    
    def analyze_sentence_thai(self, sentence: str) -> List[WordInfo]:
        """Analyze Thai sentence using PyThaiNLP"""
        if not PYTHAINLP_AVAILABLE:
            # Fallback analysis
            words = word_tokenize(sentence, engine='newmm') if PYTHAINLP_AVAILABLE else sentence.split()
            return [WordInfo(word=word, pos='NOUN', index=i) for i, word in enumerate(words) if word.strip()]
        
        # Tokenize
        words = word_tokenize(sentence, engine='newmm')
        
        # POS tagging
        try:
            pos_tags = pos_tag(words, engine='perceptron')
        except:
            pos_tags = [(word, 'NOUN') for word in words]
        
        word_infos = []
        
        for i, (word, pos) in enumerate(pos_tags):
            if word.strip() and not word.isspace():
                # Map Thai POS to our simplified categories
                pos_mapping = {
                    'NOUN': 'NOUN', 'NCMN': 'NOUN', 'NPRP': 'NOUN',
                    'PRON': 'PRON', 'PPRS': 'PRON',
                    'VERB': 'VERB', 'VACT': 'VERB', 'VSTA': 'VERB',
                    'ADJ': 'ADJ', 'ADJV': 'ADJ',
                    'ADV': 'ADV', 'ADVN': 'ADV',
                    'PREP': 'ADP',
                    'CONJ': 'CONJ', 'CCONJ': 'CONJ',
                    'INTJ': 'INTJ'
                }
                
                simplified_pos = pos_mapping.get(pos, 'NOUN')
                word_infos.append(WordInfo(
                    word=word,
                    pos=simplified_pos,
                    index=i
                ))
        
        return word_infos
    
    def analyze_sentence(self, sentence: str, language: str) -> List[WordInfo]:
        """Analyze sentence based on language"""
        if language == 'en':
            return self.analyze_sentence_english(sentence)
        else:
            return self.analyze_sentence_thai(sentence)

def main():
    st.set_page_config(
        page_title="Parts of Speech Game",
        page_icon="🎮",
        layout="wide"
    )
    
    # Initialize game and all session state variables
    if 'game' not in st.session_state:
        st.session_state.game = POSGame()
    
    if 'current_sentence' not in st.session_state:
        st.session_state.current_sentence = None
    
    if 'word_infos' not in st.session_state:
        st.session_state.word_infos = []
    
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
        
    if 'score' not in st.session_state:
        st.session_state.score = 0
        
    if 'total_words' not in st.session_state:
        st.session_state.total_words = 0
        
    if 'answers_submitted' not in st.session_state:
        st.session_state.answers_submitted = False
        
    if 'api_url' not in st.session_state:
        st.session_state.api_url = None
        
    if 'language_select' not in st.session_state:
        st.session_state.language_select = 'en'  # Default to English
        
    if 'difficulty_select' not in st.session_state:
        st.session_state.difficulty_select = 'medium'  # Default to medium
    
    # Title and description
    
    # Title
    st.title("🎮 Parts of Speech Game")
    st.markdown("### เกมจับคู่ชนิดของคำ - Word Classification Game")
    
    # Sidebar for game settings
    with st.sidebar:
        st.header("⚙️ Game Settings")
        
        # Language selection
        language = st.selectbox(
            "เลือกภาษา / Select Language:",
            options=['th', 'en'],
            format_func=lambda x: "🇹🇭 ภาษาไทย" if x == 'th' else "🇺🇸 English",
            key="language_select"
        )
        
        # Difficulty selection
        difficulty = st.selectbox(
            "เลือกระดับความยาก / Select Difficulty:",
            options=['easy', 'medium', 'hard'],
            format_func=lambda x: {
                'easy': '😊 Easy ',
                'medium': '😐 Medium ', 
                'hard': '😤 Hard '
            }[x],
            key="difficulty_select"
        )
        
        # API Configuration
        st.markdown("---")
        st.subheader("🤖 LLM API Settings")
        
        use_api = st.checkbox(
            "Use LLM API / ใช้ LLM API",
            value=False,
            help="Enable this to use LLM-generated sentences from Colab API"
        )
        
        api_url = None
        if use_api:
            api_url = st.text_input(
                "API URL:",
                placeholder="https://xxxx-xx-xx-xx-xx.ngrok-free.app",
                help="Enter the ngrok URL from your Colab notebook"
            )
            
            # Test API connection
            if api_url and st.button("🔍 Test API Connection"):
                try:
                    response = requests.get(f"{api_url}/health", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'healthy':
                            st.success("✅ API is healthy and ready!")
                            st.json(data)
                        else:
                            st.error("❌ API is not healthy")
                    else:
                        st.error(f"❌ API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        
        # Start game button
        if st.button("🚀 Start New Game / เริ่มเกมใหม่", type="primary"):
            # Generate a sentence - now returns both sentence and actual difficulty
            sentence, actual_difficulty = st.session_state.game.generate_sentence_with_llm(language, difficulty, api_url)
            
            # แสดงระดับความยากที่ได้รับจริงจาก API
            if difficulty != actual_difficulty:
                st.info(f"Difficulty level: {actual_difficulty.upper()}")
            
            # Analyze the sentence
            word_infos = st.session_state.game.analyze_sentence(sentence, language)
            st.session_state.current_sentence = sentence
            st.session_state.word_infos = word_infos
            st.session_state.game_started = True
            st.session_state.score = 0
            st.session_state.total_words = len(st.session_state.word_infos)
            
            # Reset user answers
            for word_info in st.session_state.word_infos:
                word_info.user_answer = ""
                word_info.is_correct = False
            
            st.rerun()
        
        # ไม่แสดงคะแนนและความแม่นยำในหน้า Game Setting
    
    # Main game area
    if not st.session_state.game_started:
        st.info("👈 Please select your preferences and click 'Start New Game' to begin!")
        st.info("👈 กรุณาเลือกการตั้งค่าและคลิก 'เริ่มเกมใหม่' เพื่อเริ่มต้น!")
        
        # Show POS categories reference with examples
        st.markdown("### 📚 Parts of Speech Reference / อ้างอิงชนิดของคำ")
        
        # คำอธิบายและตัวอย่างของ Parts of Speech แต่ละชนิด
        pos_examples = {
            'en': {
                'NOUN': {
                    'desc': "คำนาม - คำที่เป็นชื่อคน สถานที่ สิ่งของ หรือแนวคิด",
                    'examples': "cat (แมว), city (เมือง), happiness (ความสุข), John (จอห์น)",
                    'sentence': "The cat sleeps on the sofa. / My happiness comes from helping others."
                },
                'VERB': {
                    'desc': "คำกริยา - คำที่แสดงการกระทำหรือสถานะ",
                    'examples': "run (วิ่ง), eat (กิน), think (คิด), sleep (นอน)",
                    'sentence': "She runs every morning. / I think about you."
                },
                'ADJ': {
                    'desc': "คำคุณศัพท์ - คำที่ขยายความคำนาม",
                    'examples': "happy (มีความสุข), green (สีเขียว), beautiful (สวยงาม)",
                    'sentence': "The happy child played. / She has beautiful eyes."
                },
                'ADV': {
                    'desc': "คำวิเศษณ์ - คำที่ขยายความคำกริยา คำคุณศัพท์ หรือคำวิเศษณ์อื่น",
                    'examples': "quickly (อย่างรวดเร็ว), very (มาก), extremely (อย่างยิ่ง)",
                    'sentence': "He runs quickly. / She is very smart."
                },
                'PRON': {
                    'desc': "คำสรรพนาม - คำที่ใช้แทนคำนาม",
                    'examples': "I (ฉัน), you (คุณ), he (เขา), she (เธอ), it (มัน)",
                    'sentence': "She gave it to me. / They went to the store."
                },
                'DET': {
                    'desc': "คำนำหน้านาม - คำที่นำหน้าและกำหนดความชัดเจนของคำนาม",
                    'examples': "the (คำนำหน้านามเฉพาะเจาะจง), a, an (คำนำหน้านามไม่เฉพาะเจาะจง), this (นี้)",
                    'sentence': "The dog barked. / This book is interesting."
                },
                'ADP': {
                    'desc': "คำบุพบท - คำที่แสดงความสัมพันธ์ระหว่างคำนามกับส่วนอื่นในประโยค",
                    'examples': "in (ใน), on (บน), at (ที่), with (กับ), by (โดย)",
                    'sentence': "The cat is on the table. / She walked with her friend."
                },
                'CONJ': {
                    'desc': "คำสันธาน - คำที่เชื่อมคำ วลี หรือประโยคเข้าด้วยกัน",
                    'examples': "and (และ), but (แต่), or (หรือ), because (เพราะ)",
                    'sentence': "I like tea and coffee. / He ran because he was late."
                },
                'INTJ': {
                    'desc': "คำอุทาน - คำที่แสดงอารมณ์หรือความรู้สึกอย่างฉับพลัน",
                    'examples': "wow (ว้าว), oh (โอ้), ah (อา), ouch (โอ๊ย)",
                    'sentence': "Wow! That's amazing. / Oh, I didn't see you there."
                },
                'NUM': {
                    'desc': "คำบอกจำนวน - คำที่แสดงจำนวนหรือลำดับ",
                    'examples': "one (หนึ่ง), two (สอง), first (ที่หนึ่ง), second (ที่สอง)",
                    'sentence': "I have two cats. / She won first place in the competition."
                },
                'AUX': {
                    'desc': "คำช่วยกริยา - คำที่ช่วยคำกริยาหลักในการแสดงกาล มุมมอง หรือทัศนคติ",
                    'examples': "can (สามารถ), will (จะ), must (ต้อง), should (ควร)",
                    'sentence': "You should study. / He can swim very well."
                },
                'PART': {
                    'desc': "อนุภาค - คำที่ทำหน้าที่เฉพาะทางไวยากรณ์",
                    'examples': "to (ใน 'to go'), 's (ใน 'John's')",
                    'sentence': "I want to sleep. / That's John's book."
                }
            },
            'th': {
                'NOUN': {
                    'desc': "คำนาม - คำที่เป็นชื่อคน สถานที่ สิ่งของ หรือแนวคิด",
                    'examples': "แมว (cat), เมือง (city), ความสุข (happiness), สมชาย (personal name)",
                    'sentence': "แมวนอนอยู่บนโซฟา / ความสุขของฉันมาจากการช่วยเหลือผู้อื่น"
                },
                'VERB': {
                    'desc': "คำกริยา - คำที่แสดงการกระทำหรือสถานะ",
                    'examples': "วิ่ง (run), กิน (eat), คิด (think), นอน (sleep)",
                    'sentence': "เธอวิ่งทุกเช้า / ฉันคิดถึงคุณ"
                },
                'ADJ': {
                    'desc': "คำคุณศัพท์ - คำที่ขยายความคำนาม",
                    'examples': "มีความสุข (happy), สีเขียว (green), สวยงาม (beautiful)",
                    'sentence': "เด็กที่มีความสุขกำลังเล่น / เธอมีดวงตาที่สวยงาม"
                },
                'ADV': {
                    'desc': "คำวิเศษณ์ - คำที่ขยายความคำกริยา คำคุณศัพท์ หรือคำวิเศษณ์อื่น",
                    'examples': "อย่างรวดเร็ว (quickly), มาก (very), อย่างยิ่ง (extremely)",
                    'sentence': "เขาวิ่งอย่างรวดเร็ว / เธอฉลาดมาก"
                },
                'PRON': {
                    'desc': "คำสรรพนาม - คำที่ใช้แทนคำนาม",
                    'examples': "ฉัน (I), คุณ (you), เขา (he/she), มัน (it), พวกเขา (they)",
                    'sentence': "เธอให้มันแก่ฉัน / พวกเขาไปที่ร้านค้า"
                },
                'DET': {
                    'desc': "คำนำหน้านาม - คำที่นำหน้าและกำหนดความชัดเจนของคำนาม",
                    'examples': "นี้ (this), นั้น (that), เหล่านี้ (these), นั่น (those)",
                    'sentence': "หนังสือเล่มนี้น่าสนใจ / เด็กคนนั้นกำลังวิ่ง"
                },
                'ADP': {
                    'desc': "คำบุพบท - คำที่แสดงความสัมพันธ์ระหว่างคำนามกับส่วนอื่นในประโยค",
                    'examples': "ใน (in), บน (on), ที่ (at), กับ (with), โดย (by)",
                    'sentence': "แมวอยู่บนโต๊ะ / เธอเดินไปกับเพื่อนของเธอ"
                },
                'CONJ': {
                    'desc': "คำสันธาน - คำที่เชื่อมคำ วลี หรือประโยคเข้าด้วยกัน",
                    'examples': "และ (and), แต่ (but), หรือ (or), เพราะ (because)",
                    'sentence': "ฉันชอบชาและกาแฟ / เขาวิ่งเพราะเขามาสาย"
                },
                'INTJ': {
                    'desc': "คำอุทาน - คำที่แสดงอารมณ์หรือความรู้สึกอย่างฉับพลัน",
                    'examples': "ว้าว (wow), โอ้ (oh), อา (ah), โอ๊ย (ouch)",
                    'sentence': "ว้าว! นั่นมหัศจรรย์มาก / โอ้ ฉันไม่เห็นคุณอยู่ตรงนั้น"
                },
                'NUM': {
                    'desc': "คำบอกจำนวน - คำที่แสดงจำนวนหรือลำดับ",
                    'examples': "หนึ่ง (one), สอง (two), ที่หนึ่ง (first), ที่สอง (second)",
                    'sentence': "ฉันมีแมวสองตัว / เธอชนะที่หนึ่งในการแข่งขัน"
                },
                'AUX': {
                    'desc': "คำช่วยกริยา - คำที่ช่วยคำกริยาหลักในการแสดงกาล มุมมอง หรือทัศนคติ",
                    'examples': "จะ (will), ได้ (can), ต้อง (must), ควร (should)",
                    'sentence': "คุณควรจะศึกษา / เขาสามารถว่ายน้ำได้ดีมาก"
                },
                'PART': {
                    'desc': "อนุภาค - คำที่ทำหน้าที่เฉพาะทางไวยากรณ์",
                    'examples': "ได้, ไหม, นะ, สิ, เลย",
                    'sentence': "คุณไปได้ไหม / กินข้าวกันนะ"
                }
            }
        }
        
        pos_categories = st.session_state.game.pos_categories[language]
        
        # แสดงคำอธิบายและตัวอย่างในรูปแบบตาราง
        st.markdown("""
        <style>
        .pos-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .pos-table th, .pos-table td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        .pos-table th {
            background-color: #f0f2f6;
            font-weight: bold;
        }
        .pos-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .pos-table tr:hover {
            background-color: #e6f2ff;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # สร้างตาราง HTML 
        html_table = "<table class='pos-table'>"
        html_table += f"<tr><th>Part of Speech</th><th>Description / คำอธิบาย</th><th>Examples / ตัวอย่าง</th><th>Sentence / ประโยคตัวอย่าง</th></tr>"
        
        for pos_code, pos_name in pos_categories.items():
            if pos_code in pos_examples[language]:
                desc = pos_examples[language][pos_code]['desc']
                examples = pos_examples[language][pos_code]['examples']
                sentence = pos_examples[language][pos_code]['sentence']
                html_table += f"<tr><td><b>{pos_name}</b> ({pos_code})</td><td>{desc}</td><td>{examples}</td><td>{sentence}</td></tr>"
        
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
    
    else:
        # เพิ่มปุ่ม Back กลับไปหน้าหลัก
        if st.button("⬅️ Back to Main Menu / กลับไปหน้าหลัก"):
            st.session_state.game_started = False
            st.rerun()
        
        # Show current sentence with larger font
        st.markdown("### Current Sentence / ประโยคปัจจุบัน:")
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'><h2 style='text-align: center;'>{st.session_state.current_sentence}</h2></div>", unsafe_allow_html=True)
        
        # Initialize submit state if not exists
        if 'answers_submitted' not in st.session_state:
            st.session_state.answers_submitted = False
        
        # แสดงคำในรูปแบบตารางที่สวยงามขึ้น
        st.markdown("<h3 style='text-align: center;'>🎯 จับคู่คำกับชนิดของคำ / Match words with their parts of speech</h3>", unsafe_allow_html=True)
        
        # สร้างตารางสำหรับแสดงคำและตัวเลือก
        all_answered = True
        
        # สร้างตารางสวยงาม
        data = []
        for i, word_info in enumerate(st.session_state.word_infos):
            # Get POS options based on language
            pos_options = st.session_state.game.pos_categories[language]
            
            # Add empty option
            options = [("", "Select...")] + [(pos, name) for pos, name in pos_options.items()]
            
            # Create selectbox for POS selection
            selected = st.selectbox(
                f"**{word_info.word}**",
                options=options,
                format_func=lambda x: x[1],
                key=f"pos_select_{i}",
                disabled=st.session_state.answers_submitted  # Disable after submission
            )
            
            # Update user answer
            word_info.user_answer = selected[0] if selected else ""
            
            # Check if all words have been answered
            if not word_info.user_answer:
                all_answered = False
            
            # Check if answer is correct
            word_info.is_correct = word_info.user_answer == word_info.pos
            
            # ตรวจสอบว่า pos_name มีค่าหรือไม่ ถ้าไม่มีให้ตั้งค่าจาก pos_categories
            if not hasattr(word_info, 'pos_name') or not word_info.pos_name:
                word_info.pos_name = st.session_state.game.pos_categories[language].get(word_info.pos, word_info.pos)
                
            # เพิ่มข้อมูลในตาราง
            if st.session_state.answers_submitted:
                if word_info.is_correct:
                    result = "✅ Correct!"
                else:
                    result = "❌ Incorrect"
                    
                # เพิ่มข้อมูลคำ, คำตอบของผู้เล่น, ผลลัพธ์, และเฉลย
                data.append([word_info.word, 
                             selected[1] if selected else "", 
                             result, 
                             word_info.pos_name])
            else:
                # เพิ่มข้อมูลสำหรับกรณีที่ยังไม่ได้ส่งคำตอบ
                data.append([word_info.word, 
                             selected[1] if selected else "", 
                             "", 
                             ""])
        
        # แสดงตารางสรุปเมื่อส่งคำตอบแล้ว
        if st.session_state.answers_submitted:
            st.markdown("<h3 style='text-align: center;'>📝 สรุปผล / Results</h3>", unsafe_allow_html=True)
            df = pd.DataFrame(data, columns=["Word / คำ", 
                                           "Your Answer / คำตอบของคุณ", 
                                           "Result / ผลลัพธ์", 
                                           "Correct Answer / เฉลย"])
            
            # ใช้ตารางแบบใหม่ที่มีการจัดกึ่งกลาง
            # สร้าง CSS สำหรับตารางที่สวยงาม
            st.markdown("""
            <style>
            .table-container {
                display: flex;
                justify-content: center;
                margin: 20px 0;
            }
            .centered-df {
                width: 90%;
                margin: 0 auto;
                border-collapse: collapse;
                box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            }
            .centered-df th, .centered-df td {
                text-align: center !important;
                padding: 10px;
                border: 1px solid #ddd;
            }
            .centered-df th {
                background-color: #f0f2f6;
                font-weight: bold;
            }
            .centered-df tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .centered-df tr:hover {
                background-color: #e6f2ff;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # แปลง DataFrame เป็น HTML และใส่ class สำหรับการจัดกึ่งกลาง
            html_table = df.to_html(index=False, classes='centered-df')
            # ใส่ div ครอบตารางเพื่อจัดกึ่งกลาง
            html_table = f'<div class="table-container">{html_table}</div>'
            st.markdown(html_table, unsafe_allow_html=True)
        
        # Submit button - only show if not yet submitted and all questions answered
        if not st.session_state.answers_submitted:
            if all_answered:
                if st.button("✅ Submit Answers / ส่งคำตอบ", type="primary"):
                    st.session_state.answers_submitted = True
                    
                    # Update score
                    current_score = sum(1 for word_info in st.session_state.word_infos if word_info.is_correct)
                    st.session_state.score = current_score
                    
                    st.rerun()
            else:
                st.warning("Please answer all questions before submitting / กรุณาตอบคำถามทั้งหมดก่อนส่งคำตอบ")
        
        # Show game completion after submission
        if st.session_state.answers_submitted:
            st.markdown("---")
            st.markdown("### 🎉 Game Complete! / เกมเสร็จสิ้น!")
            
            final_score = st.session_state.score
            total_words = st.session_state.total_words
            accuracy = (final_score / total_words * 100) if total_words > 0 else 0
            
            # Show score with larger display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Score / คะแนนสุดท้าย", f"{final_score}/{total_words}")
            with col2:
                st.metric("Accuracy / ความแม่นยำ", f"{accuracy:.1f}%")
                
            # Play again button - กดแล้วสุ่มประโยคใหม่ทันที
            if st.button("🔄 Play Again / เล่นอีกครั้ง", type="primary"):
                # ใช้การตั้งค่าเดิม (ภาษาและระดับความยาก)
                language = st.session_state.language_select
                difficulty = st.session_state.difficulty_select
                api_url = st.session_state.get('api_url', None)
                
                # สร้างประโยคใหม่ทันที
                sentence, actual_difficulty = st.session_state.game.generate_sentence_with_llm(language, difficulty, api_url)
                st.session_state.current_sentence = sentence
                st.session_state.word_infos = st.session_state.game.analyze_sentence(sentence, language)
                st.session_state.score = 0
                st.session_state.total_words = len(st.session_state.word_infos)
                st.session_state.answers_submitted = False
                
                # รีเซ็ตคำตอบของผู้ใช้
                for word_info in st.session_state.word_infos:
                    word_info.user_answer = ""
                    word_info.is_correct = False
                
                st.rerun()
                
            # Show feedback based on accuracy
            if accuracy >= 80:
                st.success("🌟 Excellent!")
            elif accuracy >= 60:
                st.info("👍 Good job!")
            else:
                st.warning("💪 Keep practicing!")

if __name__ == "__main__":
    main()