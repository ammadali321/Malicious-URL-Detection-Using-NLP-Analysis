import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "../data"

st.set_page_config(page_title="Malicious URL Detector (NLP)", layout="wide", page_icon="🛡️")
st.title("🛡️ Advanced Malicious URL Detection using NLP")
st.markdown("""
**Project:** Information Security Term Project (Fall 2025)  
**Method:** Enhanced Random Forest + Advanced NLP Features + HTML Structure Analysis  
**Authors:** Muiz Ul Islam Khan, Ammad Ali, Mustafa Sajid
""")

KNOWN_LEGITIMATE_DOMAINS = {
    'facebook.com', 'google.com', 'youtube.com', 'amazon.com', 'microsoft.com',
    'apple.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'github.com',
    'stackoverflow.com', 'reddit.com', 'wikipedia.org', 'netflix.com', 'spotify.com',
    'paypal.com', 'ebay.com', 'yahoo.com', 'bing.com', 'adobe.com', 'oracle.com',
    'ibm.com', 'intel.com', 'nvidia.com', 'tesla.com', 'uber.com', 'airbnb.com',
    'dropbox.com', 'salesforce.com', 'zoom.us', 'slack.com', 'mozilla.org',
    'w3.org', 'ietf.org', 'github.io', 'medium.com', 'wordpress.com',
    'tumblr.com', 'pinterest.com', 'flickr.com', 'imgur.com', 'twitch.tv'
}

def is_known_legitimate_domain(url):
    """
    Checks if the domain is in our whitelist of known legitimate sites.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        # Check exact match and base domain
        if domain in KNOWN_LEGITIMATE_DOMAINS:
            return True
        # Check if it's a subdomain of a known legitimate domain
        for legit_domain in KNOWN_LEGITIMATE_DOMAINS:
            if domain.endswith('.' + legit_domain):
                return True
        return False
    except:
        return False

def extract_url_features(url):
    """
    Extracts comprehensive URL-based features that are strong indicators of phishing.
    """
    features = []
    try:
        parsed = urlparse(url)
        
        # URL length (phishing URLs are often longer)
        features.append(len(url))
        
        # Domain length
        domain = parsed.netloc or parsed.path.split('/')[0]
        features.append(len(domain))
        
        # Number of subdomains
        subdomains = domain.split('.')
        features.append(len(subdomains) - 2 if len(subdomains) > 2 else 0)
        
        # Has IP address (phishing sites sometimes use IPs)
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features.append(1 if re.search(ip_pattern, url) else 0)
        
        # Number of hyphens (suspicious domains often have many)
        features.append(domain.count('-'))
        
        # Number of digits in domain
        features.append(sum(c.isdigit() for c in domain))
        
        # Has HTTPS
        features.append(1 if parsed.scheme == 'https' else 0)
        
        # Path length
        features.append(len(parsed.path))
        
        # Query string length
        features.append(len(parsed.query))
        
        # Number of special characters
        special_chars = ['@', '#', '$', '%', '&', '*', '+', '=', '?', '!']
        features.append(sum(url.count(c) for c in special_chars))
        
        # Suspicious keywords in URL (but only in path/query, not domain)
        suspicious_keywords = ['login', 'secure', 'verify', 'account', 'update', 'confirm', 
                             'bank', 'paypal', 'amazon', 'ebay', 'microsoft', 'apple']
        url_lower = url.lower()
        # Only count keywords in path/query, not in domain (legitimate sites may have these in domain)
        path_query = (parsed.path + '?' + parsed.query).lower()
        features.append(sum(1 for kw in suspicious_keywords if kw in path_query))
        
        # Domain entropy (legitimate sites have lower entropy) - Index 11
        if domain:
            char_freq = {}
            for char in domain:
                char_freq[char] = char_freq.get(char, 0) + 1
            entropy = -sum((freq/len(domain)) * np.log2(freq/len(domain)) 
                          for freq in char_freq.values() if freq > 0)
            features.append(entropy)
        else:
            features.append(0)
        
        # TLD analysis - extract and classify TLD
        domain_parts_list = domain.split('.')
        extracted_tld = domain_parts_list[-1] if len(domain_parts_list) > 1 else ''
        if ':' in extracted_tld:
            extracted_tld = extracted_tld.split(':')[0]
        tld_normalized = extracted_tld.lower().strip()
        
        legitimate_tld_set = {'com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn'}
        phishing_tld_set = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'click', 'download', 'stream'}
        
        features.append(1 if tld_normalized in legitimate_tld_set else 0)
        features.append(1 if tld_normalized in phishing_tld_set else 0)
        
    except Exception as e:
        features = [0] * 14
    
    if len(features) > 14:
        features = features[:14]
    elif len(features) < 14:
        features.extend([0] * (14 - len(features)))
    
    return features

def get_enhanced_html_features(html_content):
    """
    Extracts structural and content features from HTML (28 values).
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        features = []
        
        # Basic structural features (8 features)
        features.append(len(soup.find_all('input')))
        features.append(len(soup.find_all('input', {'type': 'password'})))
        features.append(len(soup.find_all('form')))
        features.append(len(soup.find_all('script')))
        features.append(len(soup.find_all('a')))
        features.append(len(soup.find_all('img')))
        features.append(len(soup.find_all('iframe')))
        features.append(len(soup.find_all('link')))
        
        # Body content analysis (3 features)
        body = soup.body
        if body:
            body_text = body.get_text()
            body_len = len(str(body))
            text_len = len(body_text)
            features.append(body_len)
            features.append(text_len)
            features.append(text_len / body_len if body_len > 0 else 0)  # Text density
        else:
            features.extend([0, 0, 0])
        
        # Form analysis (3 features) - using already counted forms
        forms = soup.find_all('form')
        if forms:
            form_actions = [f.get('action', '') for f in forms]
            external_actions = sum(1 for action in form_actions if action.startswith('http'))
            features.append(external_actions)  # Don't duplicate form count
            features.append(external_actions / len(forms) if forms else 0)
            # Add form method analysis
            form_methods = [f.get('method', 'get').lower() for f in forms]
            features.append(sum(1 for m in form_methods if m == 'post'))
        else:
            features.extend([0, 0, 0])
        
        # Link analysis (3 features) - using already counted links
        links = soup.find_all('a', href=True)
        if links:
            external_links = sum(1 for link in links if link['href'].startswith('http'))
            features.append(external_links)
            features.append(external_links / len(links) if links else 0)
            # Add empty link count
            features.append(sum(1 for link in links if not link.get_text().strip()))
        else:
            features.extend([0, 0, 0])
        
        # Meta tags analysis (1 feature)
        meta_tags = soup.find_all('meta')
        features.append(len(meta_tags))
        
        # Suspicious patterns in HTML (1 feature) - More nuanced detection
        html_str = str(html_content).lower()
        # Only count truly suspicious patterns, not common legitimate ones
        # Phishing sites often use these in combination with other red flags
        highly_suspicious_patterns = [
            'eval(',  # Code obfuscation
            'document.write',  # Dynamic content injection
            'innerhtml',  # Often used for XSS
            'atob(',  # Base64 decoding (obfuscation)
            'unescape(',  # String obfuscation
        ]
        # Count only highly suspicious patterns, not common ones like onclick/iframe
        suspicious_count = sum(1 for pattern in highly_suspicious_patterns if pattern in html_str)
        # Also check for suspicious combinations (e.g., many iframes with few links)
        iframe_count = len(soup.find_all('iframe'))
        link_count = len(soup.find_all('a'))
        # Suspicious if many iframes but few links (common in phishing)
        if iframe_count > 3 and link_count < 5:
            suspicious_count += 1
        features.append(suspicious_count)
        
        # External resource loading (1 feature)
        scripts = soup.find_all('script', src=True)
        external_scripts = sum(1 for s in scripts if s['src'].startswith('http'))
        features.append(external_scripts)
        
        # Title and meta description (2 features)
        title = soup.find('title')
        features.append(len(title.get_text()) if title else 0)
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        features.append(len(meta_desc.get('content', '')) if meta_desc else 0)
        
        # Additional features to reach 28 total - Legitimacy indicators
        # These help distinguish legitimate sites from phishing sites
        
        # Feature 25: Proper meta tags count (legitimate sites usually have more structured meta)
        proper_meta_tags = len([m for m in meta_tags if m.get('name') or m.get('property') or m.get('http-equiv')])
        features.append(proper_meta_tags)
        
        # Feature 26: Has favicon (legitimate sites usually have)
        favicon = len(soup.find_all('link', rel=lambda x: x and 'icon' in str(x).lower()))
        features.append(favicon)
        
        # Feature 27: Content structure - paragraph count (legitimate sites have more content)
        paragraph_count = len(soup.find_all('p'))
        features.append(paragraph_count)
        
        # Feature 28: Navigation structure (legitimate sites have nav elements)
        nav_count = len(soup.find_all('nav')) + len(soup.find_all(['ul', 'ol'], class_=lambda x: x and 'nav' in str(x).lower()))
        features.append(nav_count)
        
        # Ensure exactly 28 features
        if len(features) != 28:
            # Pad or truncate to exactly 28
            if len(features) < 28:
                features.extend([0] * (28 - len(features)))
            else:
                features = features[:28]
        
        return features
        
    except Exception as e:
        # Return zeros if parsing fails - exactly 28 features
        return [0] * 28

def advanced_text_cleaning(text):
    """
    Advanced text preprocessing for better NLP analysis.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove very short words (likely noise)
    words = text.split()
    words = [w for w in words if len(w) > 2]
    
    return ' '.join(words)

def extract_text_features(text):
    """
    Extracts statistical features from text content.
    """
    if not text:
        return [0] * 10
    
    features = []
    
    # Basic statistics
    features.append(len(text))
    features.append(len(text.split()))
    features.append(len(set(text.split())))  # Unique words
    features.append(len(set(text.split())) / len(text.split()) if text.split() else 0)  # Vocabulary diversity
    
    # Character-level features
    features.append(sum(c.isdigit() for c in text))
    features.append(sum(c.isupper() for c in text))
    features.append(sum(c.isspace() for c in text))
    
    # Suspicious keywords in content - More nuanced detection
    # Only count when they appear in suspicious contexts
    text_lower = text.lower()
    # High-risk phrases (more specific, less likely in legitimate content)
    high_risk_phrases = [
        'account suspended', 'verify your account', 'click here immediately',
        'urgent action required', 'limited time offer', 'verify now',
        'account locked', 'suspended account', 'verify immediately'
    ]
    # Count high-risk phrases (more reliable than single words)
    high_risk_count = sum(1 for phrase in high_risk_phrases if phrase in text_lower)
    
    # Also check for excessive use of urgency words
    urgency_words = ['urgent', 'immediately', 'now', 'asap', 'hurry']
    urgency_count = sum(1 for word in urgency_words if word in text_lower)
    # Only count if there are multiple urgency words (legitimate sites rarely use many)
    excessive_urgency = 1 if urgency_count >= 3 else 0
    
    features.append(high_risk_count + excessive_urgency)
    
    # Average word length
    words = text.split()
    features.append(np.mean([len(w) for w in words]) if words else 0)
    
    # Exclamation/question marks (phishing often uses many)
    features.append(text.count('!') + text.count('?'))
    
    return features

def clean_html(html_content):
    """
    Enhanced HTML text extraction with better cleaning.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator=' ')
        
        # Advanced cleaning
        text = advanced_text_cleaning(text)
        
        return text
    except Exception as e:
        return ""

def tokenize_url(url):
    """
    Enhanced URL tokenization.
    """
    try:
        # Remove protocol and www
        url = url.replace("https://", "").replace("http://", "").replace("www.", "")
        
        # Remove query parameters
        url = url.split('?')[0]
        
        # Split by delimiters
        tokens = re.split(r'[./\-_?=&]', url)
        
        # Filter and clean tokens
        tokens = [t.lower() for t in tokens if t and len(t) > 2 and not t.isdigit()]
        
        return " ".join(tokens)
    except:
        return ""

def load_data(base_path):
    """
    Enhanced data loading with better error handling and feature extraction.
    """
    data = []
    benign_path = os.path.join(base_path, 'genuine_site_0')
    phish_path = os.path.join(base_path, 'phishing_site_1')
    
    if not os.path.exists(benign_path) or not os.path.exists(phish_path):
        st.error(f"Error: Data folders not found at {base_path}")
        return pd.DataFrame()
    
    benign_files = [f for f in os.listdir(benign_path) if f.endswith(('.html', '.txt'))]
    phish_files = [f for f in os.listdir(phish_path) if f.endswith(('.html', '.txt'))]
    
    # Balance dataset
    min_count = min(len(benign_files), len(phish_files))
    final_count = min(min_count, 2000)  # Reasonable limit for performance
    
    st.write(f"⚖️ Loading {final_count} Benign and {final_count} Phishing samples...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load Benign samples
    loaded_benign = 0
    for i, filename in enumerate(benign_files[:final_count * 2]):  # Try more files to account for failures
        if loaded_benign >= final_count:
            break
        try:
            filepath = os.path.join(benign_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content) > 50:  # Minimum content threshold
                    # Extract URL from filename if possible
                    url = filename.replace('.html', '').replace('.txt', '').replace('_', '.')
                    
                    # Extract all features
                    html_features = get_enhanced_html_features(content)
                    cleaned_text = clean_html(content)
                    text_features = extract_text_features(cleaned_text)
                    url_features = extract_url_features(url)
                    
                    # Validate feature dimensions
                    if len(html_features) != 28:
                        html_features = html_features[:28] if len(html_features) > 28 else html_features + [0] * (28 - len(html_features))
                    if len(text_features) != 10:
                        text_features = text_features[:10] if len(text_features) > 10 else text_features + [0] * (10 - len(text_features))
                    if len(url_features) != 14:
                        url_features = url_features[:14] if len(url_features) > 14 else url_features + [0] * (14 - len(url_features))
                    
                    data.append({
                        'text': cleaned_text,
                        'html_features': html_features,
                        'text_features': text_features,
                        'url_features': url_features,
                        'label': 0
                    })
                    loaded_benign += 1
        except Exception as e:
            continue
        
        if i % 50 == 0:
            status_text.text(f"Loading benign: {loaded_benign}/{final_count}")
            progress_bar.progress(loaded_benign / (final_count * 2))
    
    # Load Phishing samples
    loaded_phish = 0
    for i, filename in enumerate(phish_files[:final_count * 2]):
        if loaded_phish >= final_count:
            break
        try:
            filepath = os.path.join(phish_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content) > 50:
                    url = filename.replace('.html', '').replace('.txt', '').replace('_', '.')
                    
                    html_features = get_enhanced_html_features(content)
                    cleaned_text = clean_html(content)
                    text_features = extract_text_features(cleaned_text)
                    url_features = extract_url_features(url)
                    
                    # Validate feature dimensions
                    if len(html_features) != 28:
                        html_features = html_features[:28] if len(html_features) > 28 else html_features + [0] * (28 - len(html_features))
                    if len(text_features) != 10:
                        text_features = text_features[:10] if len(text_features) > 10 else text_features + [0] * (10 - len(text_features))
                    if len(url_features) != 14:
                        url_features = url_features[:14] if len(url_features) > 14 else url_features + [0] * (14 - len(url_features))
                    
                    data.append({
                        'text': cleaned_text,
                        'html_features': html_features,
                        'text_features': text_features,
                        'url_features': url_features,
                        'label': 1
                    })
                    loaded_phish += 1
        except Exception as e:
            continue
        
        if (loaded_phish + loaded_benign) % 50 == 0:
            status_text.text(f"Loading phishing: {loaded_phish}/{final_count}")
            progress_bar.progress((loaded_benign + loaded_phish) / (final_count * 2))
    
    progress_bar.empty()
    status_text.empty()
    
    if len(data) == 0:
        st.error("No data loaded! Check data files.")
        return pd.DataFrame()
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    st.success(f"✅ Loaded {len(df)} samples ({df['label'].sum()} phishing, {len(df) - df['label'].sum()} benign)")
    return df

# --- APP TABS ---
tab1, tab2, tab3 = st.tabs(["1. Train Model", "2. Model Evaluation", "3. Live Scanner"])

# GLOBAL VARIABLES
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None

# --- TAB 1: TRAIN MODEL ---
with tab1:
    st.header("Model Training Pipeline")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("📊 Dataset: 'phishing_site_1' vs 'genuine_site_0'")
        st.markdown("""
        **Enhanced Features:**
        - Advanced NLP text analysis with TF-IDF
        - 28 HTML structural features
        - 14 URL-based features (including domain entropy)
        - 10 Text statistical features
        - Feature scaling for optimal performance
        - Known legitimate domain whitelist
        """)
    
    if st.button("🚀 Load Data & Train Model", type="primary"):
        with st.spinner("Processing dataset and training model..."):
            # Load data
            df = load_data(DATA_PATH)
            
            if df.empty:
                st.error("Failed to load data!")
            else:
                st.write(f"📈 Dataset shape: {df.shape}")
                
                # 1. Text Vectorization with optimized parameters
                st.write("🔤 Vectorizing text with TF-IDF...")
                tfidf = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True  # Use logarithmic scaling
                )
                X_text = tfidf.fit_transform(df['text'])
                st.write(f"   Text features: {X_text.shape[1]} dimensions")
                
                # 2. Combine all numerical features
                st.write("🔢 Combining numerical features...")
                html_feats = np.array(df['html_features'].tolist())
                text_feats = np.array(df['text_features'].tolist())
                url_feats = np.array(df['url_features'].tolist())
                
                # Validate feature dimensions
                expected_html = 28
                expected_text = 10
                expected_url = 14  # Updated to include entropy feature
                
                # Ensure consistent dimensions
                if html_feats.shape[1] != expected_html:
                    st.warning(f"⚠️ HTML features: expected {expected_html}, got {html_feats.shape[1]}. Padding/truncating...")
                    if html_feats.shape[1] < expected_html:
                        padding = np.zeros((html_feats.shape[0], expected_html - html_feats.shape[1]))
                        html_feats = np.hstack([html_feats, padding])
                    else:
                        html_feats = html_feats[:, :expected_html]
                
                if text_feats.shape[1] != expected_text:
                    st.warning(f"⚠️ Text features: expected {expected_text}, got {text_feats.shape[1]}. Padding/truncating...")
                    if text_feats.shape[1] < expected_text:
                        padding = np.zeros((text_feats.shape[0], expected_text - text_feats.shape[1]))
                        text_feats = np.hstack([text_feats, padding])
                    else:
                        text_feats = text_feats[:, :expected_text]
                
                if url_feats.shape[1] != expected_url:
                    st.warning(f"⚠️ URL features: expected {expected_url}, got {url_feats.shape[1]}. Padding/truncating...")
                    if url_feats.shape[1] < expected_url:
                        padding = np.zeros((url_feats.shape[0], expected_url - url_feats.shape[1]))
                        url_feats = np.hstack([url_feats, padding])
                    else:
                        url_feats = url_feats[:, :expected_url]
                
                # Combine all numerical features
                X_numerical = np.hstack([html_feats, text_feats, url_feats])
                expected_numerical = expected_html + expected_text + expected_url
                st.write(f"   Numerical features: {X_numerical.shape[1]} dimensions (expected {expected_numerical})")
                
                # 3. Scale numerical features
                st.write("⚖️ Scaling numerical features...")
                scaler = RobustScaler()  # Robust to outliers
                X_numerical_scaled = scaler.fit_transform(X_numerical)
                
                # Store expected feature count for validation
                st.session_state['expected_numerical_features'] = expected_numerical
                
                # 4. Combine text and numerical features
                st.write("🔗 Combining all features...")
                X_numerical_sparse = csr_matrix(X_numerical_scaled)
                X = hstack([X_text, X_numerical_sparse])
                y = df['label']
                
                st.write(f"   Total features: {X.shape[1]} dimensions")
                
                # 5. Split data
                st.write("✂️ Splitting data (80/20)...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # 6. Train model with better hyperparameters
                st.write("🌲 Training Random Forest model...")
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=30,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                # 7. Quick validation
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                st.session_state['model'] = model
                st.session_state['vectorizer'] = tfidf
                st.session_state['scaler'] = scaler
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success(f"✅ Training Complete!")
                st.metric("Training Accuracy", f"{train_score*100:.2f}%")
                st.metric("Validation Accuracy", f"{test_score*100:.2f}%")
                
                # Feature importance (top features)
                feature_importances = model.feature_importances_
                top_indices = np.argsort(feature_importances)[-10:][::-1]
                st.write("🔝 Top 10 Most Important Features:")
                for idx in top_indices:
                    st.write(f"   Feature {idx}: {feature_importances[idx]:.4f}")

# --- TAB 2: EVALUATION ---
with tab2:
    st.header("Model Performance Evaluation")
    
    if st.session_state['model'] is not None:
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            st.metric("Accuracy", f"{acc*100:.2f}%")
        with col_metrics2:
            st.metric("AUC-ROC", f"{auc:.4f}")
        with col_metrics3:
            precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
            st.metric("Precision", f"{precision:.4f}")
        
        # Confusion Matrix and Classification Report
        col_eval1, col_eval2 = st.columns(2)
        
        with col_eval1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Benign', 'Phishing'],
                       yticklabels=['Benign', 'Phishing'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        with col_eval2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # Detailed Classification Report
        st.subheader("Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Feature importance visualization
        st.subheader("Feature Importance Analysis")
        feature_importances = model.feature_importances_
        top_20_indices = np.argsort(feature_importances)[-20:][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_importances = feature_importances[top_20_indices]
        ax.barh(range(len(top_20_indices)), top_importances)
        ax.set_yticks(range(len(top_20_indices)))
        ax.set_yticklabels([f'Feature {idx}' for idx in top_20_indices])
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importances')
        ax.invert_yaxis()
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ Please train the model in Tab 1 first.")

# --- TAB 3: LIVE SCANNER ---
with tab3:
    st.header("🔍 Live URL Scanner")
    st.markdown("Enter a URL to fetch its content and analyze it using advanced NLP and feature analysis.")
    
    url_input = st.text_input("Enter URL (e.g., https://example.com)", "", placeholder="https://example.com")
    
    col_scan1, col_scan2 = st.columns([1, 4])
    with col_scan1:
        scan_button = st.button("🔍 Scan URL", type="primary")
    
    if scan_button:
        if st.session_state['model'] is None:
            st.error("❌ Model not trained yet! Please go to Tab 1 and train the model first.")
        elif not url_input:
            st.error("❌ Please enter a URL to scan.")
        else:
            try:
                with st.spinner(f"🔍 Analyzing {url_input}..."):
                    # Fetch content
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive'
                    }
                    
                    raw_html = ""
                    clean_body_text = ""
                    html_features = [0] * 28
                    text_features = [0] * 10
                    url_features = [0] * 14
                    
                    # Check if domain is in whitelist
                    is_legitimate = is_known_legitimate_domain(url_input)
                    
                    # Try to fetch content
                    try:
                        response = requests.get(url_input, headers=headers, timeout=10, allow_redirects=True)
                        if response.status_code == 200:
                            raw_html = response.text
                            html_features = get_enhanced_html_features(raw_html)
                            clean_body_text = clean_html(raw_html)
                            text_features = extract_text_features(clean_body_text)
                        else:
                            st.warning(f"⚠️ Could not fetch content (Status: {response.status_code}). Using URL analysis only.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch content: {str(e)}. Using URL analysis only.")
                    
                    # Extract URL features
                    url_features = extract_url_features(url_input)
                    
                    # Tokenize URL
                    url_tokens = tokenize_url(url_input)
                    
                    # Combine text (URL tokens + body text)
                    combined_text = url_tokens + " " + clean_body_text
                    
                    # Vectorize text
                    vectorized_text = st.session_state['vectorizer'].transform([combined_text])
                    
                    # Validate and ensure consistent feature dimensions
                    expected_html = 28
                    expected_text = 10
                    expected_url = 14  # Updated to include entropy feature
                    expected_total = expected_html + expected_text + expected_url
                    
                    # Ensure features have correct dimensions
                    if len(html_features) != expected_html:
                        if len(html_features) < expected_html:
                            html_features.extend([0] * (expected_html - len(html_features)))
                        else:
                            html_features = html_features[:expected_html]
                    
                    if len(text_features) != expected_text:
                        if len(text_features) < expected_text:
                            text_features.extend([0] * (expected_text - len(text_features)))
                        else:
                            text_features = text_features[:expected_text]
                    
                    if len(url_features) != expected_url:
                        if len(url_features) < expected_url:
                            url_features.extend([0] * (expected_url - len(url_features)))
                        else:
                            url_features = url_features[:expected_url]
                    
                    # Ensure url_features is exactly expected_url length
                    if len(url_features) != expected_url:
                        url_features = (url_features + [0] * expected_url)[:expected_url]
                    
                    # Combine all numerical features
                    all_numerical = np.hstack([html_features, text_features, url_features])
                    
                    # Validate against scaler expectations
                    if 'expected_numerical_features' in st.session_state:
                        expected = st.session_state['expected_numerical_features']
                        if len(all_numerical) != expected:
                            st.error(f"❌ Feature mismatch: Got {len(all_numerical)} features, expected {expected}")
                            st.stop()
                    
                    # Ensure it's a 2D array for scaler
                    all_numerical = all_numerical.reshape(1, -1)
                    all_numerical_scaled = st.session_state['scaler'].transform(all_numerical)
                    numerical_sparse = csr_matrix(all_numerical_scaled)
                    
                    # Final feature vector
                    final_features = hstack([vectorized_text, numerical_sparse])
                    
                    # Predict
                    proba = st.session_state['model'].predict_proba(final_features)[0]
                    phishing_score = proba[1]
                    
                    # Apply whitelist override for known legitimate domains
                    if is_legitimate:
                        # If it's a known legitimate domain, significantly reduce the phishing score
                        phishing_score = min(phishing_score, 0.2)  # Cap at 20% for known legitimate sites
                        st.info("ℹ️ This domain is in our whitelist of known legitimate sites. Score adjusted accordingly.")
                    
                    # Use a higher threshold when content is not available (more conservative)
                    content_available = len(clean_body_text) > 100
                    threshold = 0.6 if not content_available else 0.5  # Higher threshold when only URL analysis
                    
                    prediction = 1 if phishing_score > threshold else 0
                    
                    # Display results
                    st.divider()
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        if is_legitimate and prediction == 0:
                            st.success("✅ **BENIGN / SAFE WEBSITE**")
                            st.info("ℹ️ This domain is verified in our legitimate sites whitelist.")
                            st.write("**Risk Level:** VERY LOW")
                            st.write("**Analysis:** Known legitimate website.")
                        elif prediction == 1:
                            st.error("🚨 **MALICIOUS / PHISHING DETECTED**")
                            st.write("**Risk Level:** HIGH")
                            st.write("**Recommendation:** Do not enter any personal information or credentials on this website.")
                        else:
                            st.success("✅ **BENIGN / SAFE WEBSITE**")
                            st.write("**Risk Level:** LOW")
                            if not content_available:
                                st.write("**Note:** Analysis based on URL features only (content unavailable).")
                            else:
                                st.write("**Analysis:** Content structure appears normal.")
                    
                    with col_res2:
                        st.write("**Phishing Probability Score:**")
                        st.progress(float(phishing_score))
                        st.metric("Confidence", f"{phishing_score*100:.2f}%")
                        
                        # Risk interpretation
                        if phishing_score < 0.3:
                            risk_level = "🟢 Very Low"
                        elif phishing_score < 0.5:
                            risk_level = "🟡 Low"
                        elif phishing_score < 0.7:
                            risk_level = "🟠 Medium"
                        else:
                            risk_level = "🔴 High"
                        st.write(f"**Risk Level:** {risk_level}")
                    
                    # Detailed analysis
                    with st.expander("📊 Detailed Technical Analysis", expanded=False):
                        col_tech1, col_tech2 = st.columns(2)
                        
                        with col_tech1:
                            st.write("**URL Features:**")
                            st.json({
                                "URL Length": url_features[0],
                                "Domain Length": url_features[1],
                                "Subdomains": url_features[2],
                                "Has IP": bool(url_features[3]),
                                "Hyphens": url_features[4],
                                "Digits": url_features[5],
                                "Has HTTPS": bool(url_features[6]),
                                "Suspicious Keywords": url_features[11]
                            })
                            
                            st.write("**HTML Structure Features:**")
                            st.json({
                                "Input Fields": html_features[0],
                                "Password Fields": html_features[1],
                                "Forms": html_features[2],
                                "Scripts": html_features[3],
                                "Links": html_features[4],
                                "Images": html_features[5],
                                "Iframes": html_features[6],
                                "Suspicious Patterns": html_features[20]
                            })
                        
                        with col_tech2:
                            st.write("**Text Content Features:**")
                            st.json({
                                "Text Length": text_features[0],
                                "Word Count": text_features[1],
                                "Unique Words": text_features[2],
                                "Vocabulary Diversity": f"{text_features[3]:.3f}",
                                "Phishing Keywords": text_features[7],
                                "Exclamation/Question Marks": text_features[9]
                            })
                            
                            st.write("**URL Tokens:**")
                            st.code(url_tokens if url_tokens else "None extracted")
                            
                            st.write("**Sample Content (first 300 chars):**")
                            st.code(clean_body_text[:300] if clean_body_text else "No content extracted")
                    
            except Exception as e:
                st.error(f"❌ Critical Error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
