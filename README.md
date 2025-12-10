# **Malicious URL Detection Using NLP Analysis**

## Overview
- Uses URL features (length, TLDs, entropy, hyphens, digits, HTTPS, etc.)
- Uses HTML structure (forms, scripts, links, iframes, meta tags, nav/content richness)
- Uses text signals (phishing phrases, urgency patterns)
- RandomForestClassifier for robust classification

### **Running Instructions:**



**1. Install dependencies and libraries:**



pip install -r requirements.txt



**2. Run the main code:**


cd src

python -m streamlit run main.py





### **Example Websites for Quick Testing:**



* ##### Malicious/Phishing Sites:



1. https://www.bet365365cc.com/
2. https://adobe-dynamic-inquiry-order.netlify.app/
3. https://richsnvail2cdwzf8ududchbc50xnyvpa5cjcows.surge.sh/home.html
4. https://www.robiox.com.py/users/476431298791/profile
5. https://subdomain.com/domain.php?domain=profile.co.gp
6. https://subdomain.com/domain.php?domain=facelook.shop.co



* ##### Safe/Benign Sites:



1. https://www.bbc.com
2. https://www.cnn.com
3. https://www.nytimes.com
4. https://www.vercel.com
5. https://www.netlify.com


## Model
- Estimator: RandomForestClassifier
- Features: 51 total
  - URL: 14
  - HTML: 28
  - Text: 10


***Note:*** The dataset of this project is sourced from Kaggle: https://www.kaggle.com/datasets/zackyzac/phishing-site-html-content?resource=download





