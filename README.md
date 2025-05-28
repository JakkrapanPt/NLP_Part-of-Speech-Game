# 🎮 Parts of Speech Game - เกมจับคู่ชนิดของคำ

A bilingual (Thai-English) educational game that helps users learn and practice identifying Parts of Speech using Natural Language Processing techniques.

## 🌟 Features

- **Bilingual Support**: Thai and English language support
- **Multiple Difficulty Levels**: Easy (4-6 words), Medium (6-10 words), Hard (11-18 words)
- **Interactive Gameplay**: Drag-and-drop or select-based word classification
- **Real-time Feedback**: Immediate scoring with correct answers and explanations
- **NLP Integration**: Uses spaCy for English and PyThaiNLP for Thai language processing
- **Educational Focus**: Covers 8 main parts of speech categories

## 📚 Parts of Speech Categories

1. **Noun (คำนาม)** - Names of people, places, things
2. **Pronoun (คำสรรพนาม)** - Words that replace nouns
3. **Verb (คำกริยา)** - Action or state words
4. **Adjective (คำคุณศัพท์)** - Descriptive words
5. **Adverb (คำกริยาวิเศษณ์)** - Words that modify verbs, adjectives, or other adverbs
6. **Preposition (คำบุพบท)** - Words showing relationships between other words
7. **Conjunction (คำสันธาน)** - Words that connect clauses or sentences
8. **Interjection (คำอุทาน)** - Exclamatory words

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Colab account (optional, for LLM API)

### Setup Steps

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd NLP3
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download language models**
   ```bash
   # For English processing
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

### 🤖 Optional: LLM API Setup (Google Colab)

For AI-generated sentences using Large Language Models:

1. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Open the `colab_setup.ipynb` file from this project
   - Upload the `fixed_colab_api_server_new.py` file from this project

2. **Run the Colab notebook**
   - Execute all cells in order
   - Copy the ngrok URL from the output

3. **Configure the local app**
   - In the Streamlit sidebar, check "Use LLM API"
   - Paste the ngrok URL
   - Test the connection
   - Start generating AI sentences!

## 🎮 How to Play

1. **Select Language**: Choose between Thai (🇹🇭) or English (🇺🇸)
2. **Choose Difficulty**: 
   - Easy: 4-6 words per sentence
   - Medium: 6-10 words per sentence
   - Hard: 11-18 words per sentence
3. **Start Game**: Click "Start New Game" to generate a random sentence
4. **Classify Words**: For each word in the sentence, select its correct part of speech
5. **Get Feedback**: 
   - ✅ Green for correct answers
   - ❌ Red for incorrect answers with the correct answer shown
   - 💡 Click on meaning to see word explanations
6. **View Results**: See your final score and accuracy percentage

## 🏗️ Technical Architecture

### Core Components

1. **Streamlit Frontend** (`app.py`)
   - User interface and game logic
   - Real-time interaction handling
   - Session state management
   - API integration for LLM services

2. **NLP Processing**
   - **spaCy**: English language processing
   - **PyThaiNLP**: Thai language processing
   - POS tagging and tokenization

3. **LLM API Server** (`colab_api_server.py`)
   - Flask-based REST API
   - Hugging Face Transformers integration
   - OpenAI API support (optional)
   - Ngrok tunneling for public access

4. **Game Engine**
   - Sentence analysis and word extraction
   - Answer validation and scoring
   - Difficulty level management

### Data Flow

#### Standard Mode (Local)
1. User selects language and difficulty
2. System uses predefined sample sentences
3. NLP libraries analyze sentence structure
4. Words are extracted with POS tags
5. User matches words to POS categories
6. System validates answers and provides feedback

#### LLM Mode (API Integration)
1. User enables LLM API and provides Colab URL
2. System sends generation request to Colab API
3. LLM generates contextual sentences
4. Local NLP processing analyzes generated content
5. Enhanced gameplay with AI-generated content

#### API Architecture

```
┌─────────────────┐    HTTP/REST   ┌──────────────────┐
│   Streamlit     │◄──────────────►│   Google Colab   │
│   (Local App)   │                │   (API Server)   │
└─────────────────┘                └──────────────────┘
         │                                   │
         │                                   │
    ┌────▼────┐                         ┌────▼────┐
    │  spaCy  │                         │   LLM   │
    │PyThaiNLP│                         │ Models  │
    └─────────┘                         └─────────┘
```


### Core Technologies

- **Frontend**: Streamlit (Python web framework)
- **English NLP**: spaCy with en_core_web_sm model
- **Thai NLP**: PyThaiNLP with perceptron POS tagger
- **Language Processing**: 
  - Tokenization for word separation
  - POS tagging for grammatical classification
  - Word sense disambiguation for contextual meanings

### Project Structure

```
├── app.py                            # Main Streamlit application
├── fixed_colab_api_server_new.py     # Google Colab API server
├── colab_setup.ipynb                 # Colab notebook for easy setup
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

### Key Files

- **`app.py`**: Main Streamlit application with game logic, UI, and API integration
- **`fixed_colab_api_server_new.py`**: Flask API server for LLM sentence generation (runs in Google Colab)
- **`colab_setup.ipynb`**: Jupyter notebook for easy Google Colab setup
- **`requirements.txt`**: All required Python packages for local installation
- **`README.md`**: Complete project documentation with setup instructions

### Key Components

1. **POSGame Class**: Main game logic and NLP processing
2. **WordInfo Dataclass**: Structure for storing word information
3. **Language Analyzers**: Separate methods for Thai and English processing
4. **Sample Sentences**: Built-in sentences for demo purposes
5. **Streamlit Interface**: Interactive web-based UI

## 🔧 Configuration

### Adding Custom Sentences

You can modify the `sample_sentences` dictionary in `app.py` to add your own practice sentences:

```python
self.sample_sentences = {
    'en': {
        'easy': ["Your custom English sentences here"],
        # ...
    },
    'th': {
        'easy': ["ประโยคภาษาไทยของคุณที่นี่"],
        # ...
    }
}
```

### Integrating with LLM APIs

To connect with Language Learning Models (like OpenAI GPT, Claude, etc.), modify the `generate_sentence_with_llm` method in the `POSGame` class.

## 🎯 Educational Objectives

- **Language Learning**: Improve understanding of grammatical structures
- **Interactive Education**: Learn through engaging gameplay
- **Bilingual Skills**: Practice both Thai and English grammar
- **Self-paced Learning**: Study anytime, anywhere
- **Immediate Feedback**: Learn from mistakes instantly

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **LLM Integration**: Connect to external APIs for sentence generation
2. **Enhanced UI**: Improve visual design and user experience
3. **More Languages**: Add support for additional languages
4. **Advanced Features**: User accounts, progress tracking, difficulty adaptation
5. **Mobile Optimization**: Better mobile device support

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **PyThaiNLP Installation Issues**:
   ```bash
   pip install --upgrade pythainlp
   ```

3. **Streamlit Not Starting**:
   - Check if port 8501 is available
   - Try: `streamlit run app.py --server.port 8502`

4. **Import Errors**:
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the installation steps
3. Ensure all dependencies are properly installed
4. Check that language models are downloaded correctly

---

## Create By

Jakkaphan Patan
✉️ jakkrapan.p@mail.kmutt.ac.th


Noppakorn Sorndech
✉️ noppakorn.s@mail.kmutt.ac.th


This project is part of the CPE 357 Natural Language Processing course.


**Happy Learning! สนุกกับการเรียนรู้!** 🎓✨


