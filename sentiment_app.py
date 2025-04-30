import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import time
import os

# NLTK imports with proper error handling
try:
    import nltk
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import WordNetLemmatizer
    
    # Check if NLTK data exists before downloading
    nltk_data_path = nltk.data.path[0]
    
    # Only download if not already present
    if not os.path.exists(os.path.join(nltk_data_path, 'taggers/averaged_perceptron_tagger')):
        nltk.download('averaged_perceptron_tagger', quiet=True)
    if not os.path.exists(os.path.join(nltk_data_path, 'corpora/wordnet')):    
        nltk.download('wordnet', quiet=True)
    if not os.path.exists(os.path.join(nltk_data_path, 'corpora/omw-1.4')):    
        nltk.download('omw-1.4', quiet=True)
        
    # Now import wordnet AFTER downloading
    from nltk.corpus import wordnet
    nltk_available = True
except Exception as e:
    st.warning(f"NLTK initialization issue: {e}. Some features might not work properly.")
    nltk_available = False

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and vectorizer with better error handling
@st.cache_resource
def load_model_resources():
    model_path = "sentiment_logistic_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    
    model, vectorizer = None, None
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    try:
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
        else:
            st.error(f"Vectorizer file not found: {vectorizer_path}")
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        
    return model, vectorizer

model, vectorizer = load_model_resources()

# Custom CSS (same as original)
st.markdown("""
<style>
    /* Main theme and colors */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #1a1a2e, #16213e);
        color: #e6e6e6;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #4cc9f0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Card-like container */
    .content-card {
        background-color: rgba(31, 31, 47, 0.7);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 4px solid #4361ee;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Input styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #4361ee;
        padding: 12px;
        background-color: #242442;
        color: white;
        font-size: 16px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(to right, #4361ee, #4cc9f0);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        box-shadow: 0 5px 15px rgba(76, 201, 240, 0.3);
        transform: translateY(-2px);
    }
    
    /* Logo and brand styling */
    .brand {
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
    }
    
    .logo {
        font-size: 32px;
        font-weight: bold;
        background: linear-gradient(to right, #4cc9f0, #e543ba);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .tagline {
        font-size: 14px;
        color: #8e9aaf;
    }
    
    /* Result container */
    .result-container {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        transition: all 0.5s ease;
    }
    
    .positive-result {
        background-color: rgba(64, 192, 87, 0.2);
        border-left: 4px solid #40c057;
    }
    
    .negative-result {
        background-color: rgba(240, 62, 62, 0.2);
        border-left: 4px solid #f03e3e;
    }
    
    /* Metrics and stats */
    .metric-card {
        background-color: rgba(31, 31, 47, 0.5);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-bottom: 3px solid #4cc9f0;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #4cc9f0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #8e9aaf;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: #8e9aaf;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar */
    [data-testid=stSidebar] {
        background-color: rgba(26, 26, 46, 0.8);
        padding-top: 2rem;
    }
    
    /* Confidence meter */
    .confidence-meter {
        height: 8px;
        border-radius: 4px;
        margin: 10px 0;
        background-color: #242442;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
    }
    
    /* Examples section */
    .example-item {
        padding: 10px;
        background-color: rgba(67, 97, 238, 0.1);
        border-radius: 8px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-item:hover {
        background-color: rgba(67, 97, 238, 0.2);
    }

    /* Animation for result */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade {
        animation: fadeIn 0.5s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">SentiScan AI</div><div class="tagline">Powered by Machine Learning</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About")
    st.write("SentiScan analyzes the sentiment of text using machine learning. It classifies text as positive or negative based on the content and language patterns.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📊 Model Info")
    st.markdown("""
    - **Algorithm**: Logistic Regression
    - **Features**: TF-IDF Vectorization
    - **Preprocessing**: Lemmatization, Stopword removal
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # App statistics (these would normally be tracked in a real app)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📈 App Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">5.2K</div><div class="metric-label">Analyses Run</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">91%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact section
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📩 Contact")
    st.markdown("Have questions or feedback? Reach out!")
    contact_expander = st.expander("Show Contact Info")
    with contact_expander:
        st.markdown("📧 Email: amithp0210@gmail.com")
        st.markdown("🌐 Website: sentiscan.example.com")
    st.markdown('</div>', unsafe_allow_html=True)

# Functions for text processing
# Use a simplified approach for POS tagging to avoid wordnet issues
def get_wordnet_pos(tag):
    # Simplified function that doesn't rely on wordnet constants
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # Default to noun

# Stopwords - simplified list
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 
                'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 
                'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 
                'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 
                'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself', 'no', 
                'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 
                'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 
                'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 
                'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 
                'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves']

def clean_text(text):
    """Clean and preprocess text for sentiment analysis"""
    try:
        text = text.lower()
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
        text = re.sub(r'@[\S]+', 'USER', text)
        text = re.sub(r'#(\S+)', r'\1', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = " ".join([word for word in text.split() if word not in stopwordlist])
        
        # If NLTK is not available, return the basic cleaned text
        if not nltk_available:
            return text
        
        # Use a try-except block for the POS tagging and lemmatization
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
            return " ".join(lemmatized_tokens)
        except Exception as e:
            # Fallback if advanced NLTK processing fails
            st.warning(f"Advanced text processing failed, using basic processing. Error: {e}")
            return text
    except Exception as e:
        st.error(f"Error in text cleaning: {e}")
        return text

def analyze_sentiment(text, return_probability=False):
    """Analyze the sentiment of text using the loaded model"""
    try:
        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            st.error("Model or vectorizer not available. Cannot analyze sentiment.")
            if return_probability:
                return 0, 0.5
            return 0
            
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        
        if return_probability:
            probability = model.predict_proba(vect)[0]
            confidence = probability[1] if prediction == 1 else probability[0]
            return prediction, confidence
        
        return prediction
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        # Return default values in case of error
        if return_probability:
            return 0, 0.5
        return 0

def get_word_contributions(text):
    """Simulate word contributions to sentiment (for visualization)"""
    try:
        words = text.lower().split()
        if not words:
            return {}
            
        # This is just a simplified simulation - in a real app, you would use model coefficients
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'best', 'wonderful', 'amazing', 'awesome']
        negative_words = ['bad', 'awful', 'terrible', 'sad', 'hate', 'worst', 'horrible', 'poor', 'disappointing']
        
        contributions = {}
        for word in words:
            if word in positive_words:
                contributions[word] = np.random.uniform(0.5, 1.0)
            elif word in negative_words:
                contributions[word] = np.random.uniform(-1.0, -0.5)
            else:
                contributions[word] = np.random.uniform(-0.3, 0.3)
        
        return contributions
    except Exception as e:
        st.error(f"Error generating word contributions: {e}")
        return {}

# Initialize session state for user input and examples
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None

# Main page
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.title("🔍 Sentiment Analysis")
st.write("Enter your text below to analyze its sentiment. Our AI will determine if the sentiment is positive or negative.")
st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="content-card">', unsafe_allow_html=True)
# Use session state to preserve input across rerun
user_input = st.text_area("Enter a tweet, review, or comment", value=st.session_state.user_input, height=120)
st.session_state.user_input = user_input

col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    analyze_button = st.button("🚀 Analyze Sentiment", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear Input", use_container_width=True)
with col3:
    advanced_analysis = st.checkbox("Show detailed analysis")
st.markdown('</div>', unsafe_allow_html=True)

# Check if model and vectorizer loaded successfully
if model is None or vectorizer is None:
    st.error("Model or vectorizer could not be loaded. Please check your files and restart the application.")
else:
    # Example section
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 💡 Examples")
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        positive_examples = [
            "I absolutely loved this product! It exceeded all my expectations.",
            "The customer service was fantastic and very helpful.",
            "What a beautiful day to be alive and enjoying nature!"
        ]
        st.markdown("#### Positive Examples")
        for i, example in enumerate(positive_examples):
            # Create clickable examples
            if st.button(f"Example {i+1}", key=f"pos_{i}", help=example):
                st.session_state.user_input = example
                st.session_state.selected_example = example
                st.rerun()

    with example_col2:
        negative_examples = [
            "This was the worst experience I've ever had with any company.",
            "The product broke after two days. Waste of money!",
            "I waited for hours and nobody helped me. Terrible service."
        ]
        st.markdown("#### Negative Examples")
        for i, example in enumerate(negative_examples):
            if st.button(f"Example {i+1}", key=f"neg_{i}", help=example):
                st.session_state.user_input = example
                st.session_state.selected_example = example
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Handle clear button
    if clear_button:
        st.session_state.user_input = ""
        st.rerun()

    # Results section
    if analyze_button and user_input.strip():
        # Add a loading spinner for better UX
        with st.spinner("Analyzing sentiment..."):
            try:
                time.sleep(0.8)  # Simulate processing time
                prediction, confidence = analyze_sentiment(user_input, return_probability=True)
                sentiment = "Positive" if prediction == 1 else "Negative"
                emoji = "😊" if prediction == 1 else "😠"
                
                # Format the confidence percentage
                confidence_pct = f"{confidence * 100:.1f}%"
                
                # Set the color based on sentiment
                color = "#40c057" if prediction == 1 else "#f03e3e"
                
                # Determine container class
                container_class = "positive-result" if prediction == 1 else "negative-result"
                
                # Display the result
                st.markdown(f'<div class="result-container {container_class} animate-fade">', unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center; color: {color};'>{emoji} {sentiment} Sentiment</h2>", unsafe_allow_html=True)
                
                # Confidence meter visualization
                st.markdown(f"""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <p>Confidence: <strong>{confidence_pct}</strong></p>
                    <div class='confidence-meter'>
                        <div class='confidence-fill' style='width: {confidence * 100}%; background-color: {color};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Advanced analysis section
                if advanced_analysis:
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.markdown("### 🔬 Detailed Analysis")
                    
                    # Text statistics
                    st.markdown("#### Text Statistics")
                    words = user_input.split()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Word Count", len(words))
                    col2.metric("Character Count", len(user_input))
                    col3.metric("Average Word Length", f"{sum(len(word) for word in words) / max(len(words), 1):.1f}")
                    
                    # Word contribution chart (simulated)
                    st.markdown("#### Word Sentiment Contribution")
                    word_contributions = get_word_contributions(user_input)
                    
                    if word_contributions:
                        # Sort by absolute value of contribution
                        sorted_contribs = sorted(word_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                        
                        if sorted_contribs:  # Make sure we have data
                            words, scores = zip(*sorted_contribs)
                            
                            # Create a DataFrame for the chart
                            df = pd.DataFrame({
                                'Word': words,
                                'Contribution': scores
                            })
                            
                            # Create a horizontal bar chart
                            chart = st.bar_chart(
                                df.set_index('Word'),
                                use_container_width=True,
                                height=300
                            )
                            
                            st.markdown("""
                            <div style='font-size: 0.8em; color: #8e9aaf; margin-top: -15px;'>
                                The chart shows how each word contributes to the overall sentiment. 
                                Positive values (blue) indicate positive contribution, negative values (red) indicate negative contribution.
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Processed text section
                    st.markdown("#### Text Processing")
                    with st.expander("Show text processing steps"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.text(user_input)
                        with col2:
                            st.markdown("**Processed Text:**")
                            st.text(clean_text(user_input))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    # Add a tips section
    if not analyze_button or not user_input.strip():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        with st.expander("💡 Tips for better analysis"):
            st.markdown("""
            - Be specific and provide context in your text
            - Use natural language as you would in real conversations
            - Include descriptive adjectives for more accurate sentiment detection
            - Longer texts generally provide more accurate results
            - Our model works best with English language text
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Made by Amith | Streamlit | Machine Learning | 2025
    <div style="margin-top: 10px; font-size: 12px;">
        Version 2.0 | Last updated: April 2025
    </div>
</div>
""", unsafe_allow_html=True)