import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import json



# Set the app title and favicon
st.set_page_config(
    page_title="Hasnain Imtiaz - Portfolio",
    page_icon="📊",
    layout="wide",
)

# Add CSS styling for the entire app
st.markdown(
    """
    <style>
    /* Reset default margin and padding */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #222; /* Dark background color */
        margin: 0;
        padding: 0;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #000 0%, #111 100%); /* Gradient background for sidebar */
        color: white; /* Text color in sidebar */
        padding: 20px; /* Add padding to the sidebar content */
    }

    /* Centered content styles */
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        text-align: center;
    }

    /* Custom button style */
    .custom-button {
        background-color: #ff5733; /* Change the background color */
        color: white; /* Change the text color */
        padding: 12px 24px; /* Adjust padding as needed */
        border: none;
        border-radius: 8px; /* Change the border radius */
        cursor: pointer;
        transition: background-color 0.3s ease-in-out;
    }
    
    /* Change button color on hover */
    .custom-button:hover {
        background-color: #cc4222; /* Change the background color on hover */
    }

    /* Add more custom styles as needed */
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a cover image
background_image = Image.open("brain.jpg")
st.image(background_image, use_column_width=True, caption="")

# Create a container for the header
header_container = st.container()

# Header
with header_container:
    # Profile picture on the left
    col1, col2 = st.columns([1, 4])
    with col1:
        profile_image = Image.open("sakib.jpeg")

        st.image(profile_image, width=220)


    # About Me section on the right
    with col2:
        st.markdown(
            "<h4 style='font-style: italic; font-weight: bold; font-family: cursive;'>In the depths of data, where patterns hide and insights slumber, I emerge as the seeker, deciphering the enigma of Machine Learning.</h4>",
            unsafe_allow_html=True)
        st.title("Hasnain Imtiaz")
        st.markdown("<i>Machine Learning, Data and Technology Enthusiast</i>", unsafe_allow_html=True)
        st.write(
            """ I possess a robust grasp of statistical concepts and command a strong understanding of cutting-edge 
            machine learning algorithms, including but not limited to Machine Learning (Supervised and Unsupervised).
             Moreover, as an independent Python programmer and developer, I've honed my skills through self-initiated projects. This hands-on 
             experience allows me to leverage Python's diverse libraries and tools proficiently, enhancing my capabilities in data analysis and model development.
             I possess both knowledge and hands-on experience, which I've gained through independent work by applying advanced methods 
             like ensembling learning, transfer learning, neural network and deep learning with a strong emphasis on comprehensive 
             data preprocessing, feature engineering, and hyperparameter tuning to maximize and elevate model performance. 
             My commitment to excellence extends to the field of Data Science and Big Data Technology, prioritizing efficient computational processes. 
 """
        )



# Interest Zones section
st.title("Interest Zones")

# Define your interest zones and keywords
interest_zones = {
    "Communication Data Science": [
            "Social Media Analytics",
            "Text/Opinion Mining",
            "Media Content Analysis",
            "Segmentation Analytics",
            "Network Analysis",
            "Sentiment Analysis"
        ],

    "Natural Language Processing": ["Word Embedding", "TF-IDF (Term Frequency-Inverse Document Frequency)", "Vectorization", "Bag of Words", "Name Entity Recognition(NER)", "Stemming/Lemmatization", "Syntactic Parsing", "Dependency Parsing",  "Topic Modeling", "Document-Term Matrix", "Text Classification", "Performance Metrics"],
    "Machine and Deep Learning": ["""Ensemble Methods and Learning (LightLGB, XGBoost, AdaBoost, GBM, Bagging, Boosting), 
                                  Neural Networks (CNN, RNN, LSTM),
                                  Transfer Learning,
                                  Sequence Labeling,
                                  Attention Mechanisms,
                                  Supervised Learning (Classification and Regression),
                                  Unsupervised Learning (Clustering),
                                  Feature Engineering,
                                  Overfitting and Regularization,
                                  Model Evaluation and Validation"""],

    "Data Visualization": ["Data Insights", "Interactive Charts", "Data Storytelling"],
}

st.write("Here are some of my key interest zones:")

# Display interest zones and keywords in keyword-like boxes
for zone, keywords in interest_zones.items():
    st.subheader(zone)
    keyword_box = ", ".join([f"`{keyword}`" for keyword in keywords])
    st.markdown(f"Keywords: {keyword_box}")




################################

# CSS for the button style
button_style = """
    background: linear-gradient(45deg, #FFD700, #FF1493, #00BFFF, #32CD32);
    background-size: 200% 200%;
    animation: glowing 3s linear infinite;
    color: white;
    padding: 20px 40px;
    font-size: 24px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease-in-out;
}

@keyframes glowing {
    0% {
        background-position: 0 0;
    }
    100% {
        background-position: 200% 200%;
    }
}

/* Add hover effect */
.big-button:hover {
    transform: scale(1.05);
}

/* Style for the big box */
.big-box {
    padding: 20px;
    background-color: #f0f0f0;
    border-radius: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
}

#surprise-box {
    display: none;
}

function throwEmoticons() {
    const surpriseBox = document.getElementById("surprise-box");
    surpriseBox.innerHTML = "🎉😄🥳";
    surpriseBox.style.display = "block";
}
"""

# Create a div with a custom button style using st.markdown
if st.button("Click Me for INSIGHTS!", key="click_me_button", help="Click to show the big box"):
    st.markdown(
        f'<div class="big-button" onclick="throwEmoticons()">'
        '<button class="stButton">Big Box</button>'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("My Perspectives on Using Communication Data Science, Machine Learning and Computations")

    st.write(
        """
        I am profoundly enthusiastic about the intrinsic processes of how information propagates through a network that can help identify key 
        influencers or nodes affecting opinions, behavior, or decisions. Ethically harnessed, this understanding becomes an operational cornerstone for catalyzing positive behavioral 
        shifts—ranging from the cultivation of health-conscious routines and altruistic tendencies to the fortification of cybersecurity measures. Moreover, its application extends to the discernment of vulnerabilities within societal cohorts or individual entities,
        as well as the systematic identification and mitigation of detrimental misinformation within virtual domains. My undergraduate studies, focused on the methodological and theoretical aspects of 
        Digital Media, Internet and Social Netowrk Analysis kindled my fascination with the notion of employing technology to forecast human conduct 
        within online systems, instilling a deep-seated desire to expand my knowledge in this captivating domain.
"""
    )

    st.write("1. **Enhancing Decision-Making:**:  In today's interconnected world, decisions made by individuals, businesses, "
             "and governments are increasingly influenced by digital communication. Communication Data Science enables "
             "us to make informed choices by providing real-time insights into public opinion, market trends, and "
             "emerging issues.")
    # Paragraph 1
    st.write("2. **Audience Analysis**: Data technology enables precise audience analysis by leveraging data "
             "analytics. It helps understand demographics, behaviors, and preferences, allowing tailored and "
             "effective communication strategies. Sentiment analysis, powered by AI and NLP, gauges public "
             "opinion, guiding strategic responses.")

    # Paragraph 2
    st.write("3. **Content Optimization**: Data-driven content optimization involves A/B testing to experiment "
             "with elements and personalize content recommendations. Machine learning enhances user engagement "
             "by delivering tailored content, boosting conversions.")

    # Paragraph 3
    st.write("4. **Media Planning and Buying**: Data science guides media planning with insights from historical "
             "and real-time data. Programmatic advertising automates ad space purchase, ensuring targeted campaigns "
             "and cost-efficient spending.")

    # Paragraph 4
    st.write("5. **Crisis Management**: Data analytics helps monitor social media and news for early crisis signs. "
             "Swift, data-informed responses mitigate damage, protecting brand reputation and integrity.")

    # Paragraph 5
    st.write("6. **Improving Public Health:**: PDuring global health crises, such as the COVID-19 pandemic, communication data "
             "plays a pivotal role in monitoring and responding to the situation. Analyzing communication patterns helps "
             "public health officials track the spread of the virus, identify misinformation, and plan effective interventions.")

    # Paragraph 5
    st.write("7. **Predictive Analytics**: Predictive analytics forecasts trends and audience behavior. It empowers "
             "strategic decision-making and resource allocation, maximizing communication impact. By analyzing communication data, "
             "we can anticipate societal trends and shifts in public sentiment. This knowledge is invaluable for businesses, "
             "policymakers, and researchers seeking to adapt to changing circumstances.")



    st.markdown('<div id="surprise-box"></div>', unsafe_allow_html=True)









###############################################


    # Load your Lottie JSON animation
    with open("animation_lnrarq3q.json", "r") as json_file:  # Replace with the path to your Lottie JSON file
        lottie_json = json.load(json_file)

    with open("animation_lmnn38qn.json", "r") as json_file2:  # Replace with the path to your second Lottie JSON file
        lottie_json2 = json.load(json_file2)

    with open("animation_lmnncvkz.json", "r") as json_file3:  # Replace with the path to your third Lottie JSON file
        lottie_json3 = json.load(json_file3)

    # Create columns to display animations
    col1, col2, col3 = st.columns(3)

    # Display the first animation in the first column
    with col1:
        st_lottie(lottie_json, speed=1, height=200)


    # Display the second animation in the second column
    with col2:
        st_lottie(lottie_json2, speed=1, height=200)


    # Display the third animation in the third column
    with col3:
        st_lottie(lottie_json3, speed=1, height=200)

# Create separate containers for the left and right sides
left_container, right_container = st.columns(2)

# Left Side (with bullet points)
with left_container:
    skills = [
        {"name": "Machine Learning (Supervised & Unsupervised)", "icon": "🤖"},
        {"name": "Natural Language Processing", "icon": "📚"},
        {"name": "Large Language Models (LLM)", "icon": "🧮"},
        {"name": "Time Series Analysis", "icon": "⏰ "},
        {"name": "Data Manipulation & Preprocessing", "icon": "🔍"},
        {"name": "Web Scraping", "icon": "🌐"},
        {"name": "Data Analysis", "icon": "📊"},
        {"name": "Python Programming", "icon": "🐍"},
        {"name": "Data Structures and Algorithms", "icon": "📂"},
        {"name": "Feature Engineering", "icon": "🛠️"},
        {"name": "Exploratory Analysis", "icon": "🔍"},
        {"name": "Data Visualization", "icon": "📈"},
    ]

    # Left container
    with st.container():
        st.header("Computational Skills")
        st.text("Here are some of the skills I've developed:")
        for skill in skills:
            st.markdown(f"- {skill['icon']} {skill['name']}")


# Right Side (with bullet points and icons)
with right_container:
    st.header("Languages and Tools")

    languages_and_tools = [
        {
            "name": "C",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/c/c-original.svg",
        },
        {
            "name": "C++",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg",
        },
        {
            "name": "Django",
            "image_url": "https://cdn.worldvectorlogo.com/logos/django.svg",
        },
        {
            "name": "Git",
            "image_url": "https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg",
        },
        {
            "name": "Hadoop",
            "image_url": "https://www.vectorlogo.zone/logos/apache_hadoop/apache_hadoop-icon.svg",
        },
        {
            "name": "Apache Spark",
            "image_url": "https://www.vectorlogo.zone/logos/apache_spark/apache_spark-icon.svg",
        },
        {
            "name": "Matlab",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/2/21/Matlab_Logo.png",
        },
        {
            "name": "MongoDB",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/mongodb/mongodb-original-wordmark.svg",
        },
        {
            "name": "MySQL",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg",
        },
        {
            "name": "OpenCV",
            "image_url": "https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg",
        },
        {
            "name": "Pandas",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg",
        },
        {
            "name": "PostgreSQL",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original-wordmark.svg",
        },
        {
            "name": "Python",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg",
        },
        {
            "name": "PyTorch",
            "image_url": "https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg",
        },
        {
            "name": "Scikit-Learn",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg",
        },
        {
            "name": "Seaborn",
            "image_url": "https://seaborn.pydata.org/_images/logo-mark-lightbg.svg",
        },
        {
            "name": "TensorFlow",
            "image_url": "https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg",
        },
        {
            "name": "R Programming",
            "image_url": "https://www.r-project.org/logo/Rlogo.png",
        },
        {
            "name": "NumPy",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg",
        },
        {
            "name": "Flask",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/flask/flask-original-wordmark.svg",
        }

    ]

    # Display the icons and names in rows of 5
    for i in range(0, len(languages_and_tools), 5):
        row = languages_and_tools[i:i + 5]
        cols = st.columns(5)
        for j, item in enumerate(row):
            with cols[j]:
                st.markdown(
                    f"<a href='#' target='_blank' rel='noreferrer'><img src='{item['image_url']}' alt='{item['name']}' width='40' height='40'/></a>",
                    unsafe_allow_html=True,
                )
                st.text(item["name"])

    # Create a box for images and badges
st.markdown("<div style='border: 2px solid #3366cc; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)

# Create a single row to display the images horizontally
col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed


# Close the box
st.markdown("</div>", unsafe_allow_html=True)

##########################

# Load your Lottie JSON animations
with open("data2.json", "r") as json_file1:  # Replace with the path to your first Lottie JSON file
    lottie_json1 = json.load(json_file1)

with open("AI.json", "r") as json_file2:  # Replace with the path to your second Lottie JSON file
    lottie_json2 = json.load(json_file2)

with open("data.json", "r") as json_file3:  # Replace with the path to your third Lottie JSON file
    lottie_json3 = json.load(json_file3)

# Create columns to display animations
col11, col22, col33 = st.columns(3)

# Display the first animation in the first column
with col11:
    st_lottie(lottie_json1, speed=1, height=200)

# Display the second animation in the second column
with col22:
    st_lottie(lottie_json2, speed=1, height=200)

# Display the third animation in the third column
with col33:
    st_lottie(lottie_json3, speed=1, height=200)
# Center the Work Experience section
# Center the Work Experience section and underline it
# Centered Work Experience
# Define a CSS style for the boxes
# Define a CSS style for the line with designs
box_style = """
    border: 2px solid #3366cc;
    background-color: #f8f8f8;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 5px 5px 10px #888888;
"""

line_style = """
    margin-top: 20px;
    margin-bottom: 20px;
    border: none;
    height: 1px;
    background: linear-gradient(to right, #3366cc, #f8f8f8, #3366cc);
"""

# Define a CSS style for the title
title_style = """
    text-align: center;
    font-size: 30px;
    color: #3366cc;
    margin-top: 10px;
    margin-bottom: 20px;
"""

# Centered Work Experience
st.markdown(f"<hr style='{line_style}'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Work Experience</h1>", unsafe_allow_html=True)
st.markdown(f"<hr style='{line_style}'>", unsafe_allow_html=True)

# Student, Machine Learning Engineering
st.markdown("<h3 style='text-align: center;'>Student, Machine Learning Engineering</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Creative IT Institute, Professional IT Certificate</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>January 2023 - Present</h4>", unsafe_allow_html=True)
# Apply box style to the description
st.markdown(f"<div style='{box_style}'>"
    "This Professional Course dives into the theoretical and practical aspects of programming concepts in Python and "
    "explores a wide range of topics in data science. We cover Python basics, object-oriented programming, and "
    "advanced data structures. This course introduced different libraries: Pandas, Numpy, Matplotlib, and Seaborn, "
    "Scikit-learn, OpenCV, PyTorch, for data manipulation, numerical computing, and visualization. Exploratory "
    "Data Analysis (EDA) is done using real-world datasets like Iris and Haberman. Probability theory, including "
    "Poisson and Gaussian distributions, is covered. NLP techniques like BOW, TF-IDF, and tokenization are introduced. "
    "Machine Learning algorithms such as KNN, Naive Bayes, logistic regression, SVM, decision trees, and random forests "
    "are explored. Clustering, sentiment analysis, recommendation systems, and cancer prediction projects are undertaken. "
    "This course equips us with theoretical knowledge and practical skills in Python, data science, and machine learning."
    "</div>", unsafe_allow_html=True)

# Research Assistant
st.markdown("<h3 style='text-align: center;'>Research Assistant</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Microgovernance Research Initiatives</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>July 2021 - December 2021</h3>", unsafe_allow_html=True)

# Apply box style to the description
st.markdown(f"<div style='{box_style}'>"
    "During my tenure as a research assistant at MGR, a renowned public university, I had the opportunity to engage in "
    "a range of valuable experiences and training programs. One of the highlights was receiving training in advanced "
    "research techniques, which equipped me with the necessary skills to conduct high-quality research. Through workshops "
    "and seminars, I gained a deeper understanding of various research methodologies and data analysis techniques, "
    "allowing me to contribute effectively to research projects. Moreover, I had the privilege of participating in "
    "workshops conducted by prominent organizations such as the International Foundation for Electoral Systems (IFES) "
    "and BRAC. These workshops focused on important societal issues, with a particular emphasis on gender, society, "
    "misinformation, cyber threats, and social media platforms. Engaging in these workshops provided me with a "
    "comprehensive understanding of these topics and their impact on contemporary society. I gained insights into the "
    "challenges posed by gender inequality, the spread of misinformation in the digital age, the growing threat of "
    "cyber attacks, and the influence of social media platforms on public opinion and discourse."
    "</div>", unsafe_allow_html=True)

# Projects
# Projects


# Custom CSS for styling
st.markdown(
    """
    <style>
    .header-text {
        font-size: 36px;
        color: #3366cc;
        text-align: center;
        padding: 20px 0;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .projects-text {
        font-size: 18px;
        color: #444;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display header and text
st.markdown('<p class="header-text">Course Works / Independent Projects</p>', unsafe_allow_html=True)
st.markdown('<p class="projects-text">Here are some of the projects I have worked on during my courseworks with real-life data:</p>', unsafe_allow_html=True)


# Create a container for the project cards
project_container = st.container()


# Function to display the detailed project description inside a markdown container
import streamlit as st

def display_detailed_description():
    st.subheader("***Game Data Analytics using Ratings, Playtime, Suggestions***")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        margin-left: 40px;  /* Adjust the margin for right shift */
    ">
    <ul>
        <li>The project aims to gain insights into the factors influencing game ratings.</li>
        <li>It creates predictive models using machine-learning approaches.</li>
        <li>It combines the RAWG Video Games Database API with a Python application to collect data about video games.</li>
        <li>Two models, Decision Trees, and Random Forests, are trained on this data.</li>
        <li>The models uncover the relationships between specific features and game ratings.</li>
        <li>Model performance is assessed using Mean Squared Error (MSE) and R-squared (R2) scores.</li>
        <li>Feature importance analysis is conducted to identify influential factors.</li>
        <li>Regression analysis results, including R-squared and Adj. R-squared values, are presented.</li>
    </ul>
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display the images in columns
    col1, col2 = st.columns(2)  # Adjust the number of columns as needed

    with col1:
        st.image("relations-RF.png", width=400, caption="Sentiment Analysis Example")
        # Add more images with st.image as needed

    with col2:
        # Add more images or graphs in the second column
        pass  # Use st.image to add images or st.pyplot to display graphs

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/your-username/your-repository)", unsafe_allow_html=True)

# Collapsible section for the project
with st.expander("Project 1: Game Data Analytics using Ratings, Playtime, Suggestions"):
    st.write("Short Description: The project aims to analyze game ratings using machine learning.")
    display_detailed_description()



# Function to display detailed description of Project 2
def display_detailed_description_project_2():
    st.subheader("***Laptop Prices: Unveiling the Key Factors Influencing Pricing Patterns***")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        margin-left: 40px;  /* Adjust the margin for right shift */
    ">
    <ul>
        <li>Description: In today's fast-paced digital world, laptops have become an indispensable tool for both work and leisure. 
        With an overwhelming variety of options available in the market, understanding the factors that determine laptop prices is crucial 
        for making informed purchasing decisions.</li>
        <li>This project delves into the world of laptop pricing, leveraging data analysis and machine learning techniques to uncover 
        the key factors that influence laptop prices.</li>
        <li>By analyzing a comprehensive dataset of laptop specifications and prices, this project aims to provide valuable insights 
        into the pricing patterns of laptops. Through the utilization of regression models, particularly Random Forest Regression, 
        we explore the relationships between various laptop features and their impact on pricing. The project's findings empower consumers 
        to make informed decisions based on their budget, performance requirements, and desired features.</li>
        <li>Furthermore, feature importance analysis is conducted to identify the most influential factors driving laptop prices. 
        This analysis allows us to uncover which features have the greatest impact on pricing decisions. The results are visualized through 
        informative graphs, highlighting the relative importance of each feature and enabling users to prioritize their preferences 
        based on budget and requirements.</li>
    </ul>
    <img src="project02.png" alt="Graph 1" style="width: 400px; margin-top: 20px;">
    <img src="laptop.png" alt="Graph 2" style="width: 400px; margin-top: 20px;">
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/your-username/your-repository)", unsafe_allow_html=True)

# Collapsible section for Project 2
with st.expander("Project 2: Laptop Prices: Unveiling the Key Factors Influencing Pricing Patterns"):
    st.write("Short Description: Understanding factors influencing laptop prices.")
    display_detailed_description_project_2()


# Function to display detailed description of Project 3
def display_detailed_description_project_3():
    st.subheader("""***Looking Through a National News Article Texts to learn about the name 'Joe Biden' !***""")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 18px;
        text-align: left;
        margin-left: auto; /* Auto margin for center alignment */
        margin-right: auto; /* Auto margin for center alignment */
    ">
    Description: The aim of this project was to leverage Natural Language Processing (NLP) techniques to analyze a large corpus 
    of national news articles to track and understand the mentions of the name 'Joe Biden.' The project incorporated various NLP 
    methods, including Word Cloud generation, handling stop words, label encoding, and creating a Bag of Words representation.
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display the images horizontally
    st.image("g20.png", width=400, caption="Sentiment Analysis Example")
    st.image("g20-1.png", width=400, caption="Image Classification Example")
    st.image("g20-3.PNG", width=400, caption="Image Classification Example")

    # Keywords section for Project 3
    keywords_project3 = [
        "National News Articles",
        "Natural Language Processing",
        "NLP Techniques",
        "Word Cloud",
        "Stop Words",
        "Label Encoding",
        "Bag of Words",
        "Joe Biden Mentions",
    ]

    keyword_boxes_project3 = [
        f'<span style="background-color: #000; color: #3366cc; padding: 2px 4px; border-radius: 4px; margin-right: 5px;">{keyword}</span>'
        for keyword in keywords_project3
    ]

    keywords_html_project3 = " ".join(keyword_boxes_project3)

    # Center-aligned Keywords for Project 3
    st.markdown(
        f"""
        <div style="text-align: center;">
            <strong>Keywords:</strong> {keywords_html_project3}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/your-username/your-repository)", unsafe_allow_html=True)

# Collapsible section for Project 3
with st.expander("Project 3: Looking Through a National News Article Texts to learn about the name 'Joe Biden'"):
    st.write("Short Description: Analyzing national news articles for 'Joe Biden' mentions.")
    display_detailed_description_project_3()


# Project 4: GamersHub: Text and Opinion Mining of Reddit Comments

def display_detailed_description_project():
    st.subheader("***GamersHub: Text and Opinion Mining of Reddit Comments***")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        margin: 20px auto; /* Auto margin for center alignment */
        max-width: 800px; /* Limiting width for better readability */
    ">
    Description: GamersHub: Text and Opinion Mining of Reddit Comments

    In the initial stages, the dataset of Reddit comments is likely collected and preprocessed. This entails tasks such as data cleaning,
    removal of irrelevant symbols, and potentially splitting the data into training and testing sets using `train_test_split` from
    `sklearn.model_selection`. In this case, the CSV file contains 21,821 rows and 3 columns. Each row in the CSV file likely represents a
    separate record or data point, while each of the 3 columns likely holds different attributes, variables, or fields associated with the data.

    To convert the text data into a format suitable for analysis, the project employs the `CountVectorizer` and `TfidfVectorizer` from
    `sklearn.feature_extraction.text`. These techniques transform the comments into numerical vectors, enabling machine learning algorithms
    to process and analyze the text data effectively.

    For a deeper understanding of the underlying themes within the comments, the project utilizes `LatentDirichletAllocation` from
    `sklearn.decomposition` to perform topic modeling. This aids in uncovering latent topics or patterns within the discussions, contributing
    valuable insights into prevalent themes.
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display the images in columns
    st.subheader("Images:")
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed
    with col1:
        st.image("reddit2.png", width=250, caption="Image 1")
    with col2:
        st.image("reddit1.png", width=250, caption="Image 2")
    with col3:
        st.image("reddit3.png", width=250, caption="Image 3")

    # Keywords section for Project
    keywords_project = [
        "Reddit Comments",
        "Text Mining",
        "Opinion Mining",
        "Data Cleaning",
        "Data Preprocessing",
        "Topic Modeling",
        "CountVectorizer",
        "TfidfVectorizer",
        "LatentDirichletAllocation",
    ]

    keyword_boxes_project = [
        f'<span style="background-color: #f0f0f0; color: #3366cc; padding: 5px 10px; border-radius: 10px; margin-right: 10px;">{keyword}</span>'
        for keyword in keywords_project
    ]

    keywords_html_project = " ".join(keyword_boxes_project)

    # Center-aligned Keywords for Project
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <strong>Keywords:</strong> {keywords_html_project}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/sakib4535/Text-Mining-and-Opinion-Reddit/tree/main)", unsafe_allow_html=True)


# Collapsible section for Project
with st.expander("Project 4: GamersHub: Text and Opinion Mining of Reddit Comments"):
    st.write("Short Description: The project aims to analyze game ratings using machine learning.")
    display_detailed_description_project()


def display_detailed_description_project_5():
    st.subheader(
        "*** Multi-Label Text Sentiment Classification and Word Embedding Analysis for the Review of 'Thread' Social Media Application")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        margin-left: auto; /* Auto margin for center alignment */
        margin-right: auto; /* Auto margin for center alignment */
    ">
    Description: Multi-Label Text Sentiment Classification and Word Embedding Analysis for the Review of 'Thread' Social Media Application
    <ul>
        <li>📢 Comprehensive Dataset: With over 37,000 reviews, our meticulously curated dataset captures a substantial and representative sample of user feedback from both Android and iOS platforms, 
        offering a comprehensive view of the "Thread" app's reception.</li>
        <li>💬 Label-wise accuracies are impressive, with Positive Sentiment at 88.25%, Negative Sentiment at 98.20%, and 
        Neutral Sentiment at 91.00%. The mean accuracy of 92.48% reflects the model's strong performance across all categories.</li>
        <li>
            <img src="your_image_path_here.png" alt="Your Image" style="width: 300px; margin-top: 10px;">
            <p style="font-style: italic; font-size: 12px; text-align: center;">Image 1: Your Image Caption</p>
        </li>
        <!-- Include more list items or images as needed -->
    </ul>

    <!-- Keywords section -->
    <div style="text-align: center; margin-top: 20px;">
        <strong>Keywords:</strong>
        <span style="background-color: #000; color: #3366cc; padding: 2px 3px; border-radius: 3px; margin-right: 5px;">Keyword1</span>
        <span style="background-color: #000; color: #3366cc; padding: 2px 3px; border-radius: 3px; margin-right: 5px;">Keyword2</span>
        <!-- Add more keywords in a similar format -->
    </div>

    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown


# Collapsible section for Project 5
with st.expander(
        "Project 5: Multi-Label Text Sentiment Classification and Word Embedding Analysis for the Review of 'Thread' Social Media Application"):
    st.write(
        "Short Description: Multi-label sentiment analysis and word embedding analysis for a social media application.")
    st.write("Click the button below to view detailed project description and analysis.")
    if st.button("Show Detailed Description 5", key="detailed_description_5"):
        display_detailed_description_project_5()


import streamlit as st

def display_detailed_description_project_6():
    st.subheader("***Telecommunication Customer Churn Prediction***")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        margin-left: 40px;  /* Adjust the margin for right shift */
    ">
    📊 Successfully conducted an in-depth analysis of a telecommunications company's customer churn dataset to enhance decision-making processes. 📈

    💡 Leveraged feature engineering techniques to extract meaningful insights from customer data. Uncovered valuable patterns that contribute to customer churn and retention. 🧠

    🤖 Employed state-of-the-art machine learning methodologies to predict customer churn and develop strategies for retention. 📉

    📈 Achieved an impressive 80% accuracy rate in predicting customer churn through an Artificial Neural Network model. Demonstrating that proactive retention strategies can significantly impact customer loyalty and company profitability. 💼

    🔍 Explored the relationship between customer characteristics and churn, identifying factors such as contract duration and service usage that influence customer decisions. 📝

    🔮 Provided actionable recommendations for improving customer retention, including the development of personalized offers, customer support enhancements, and optimized billing strategies. 💼
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/sakib4535/Telecommunication-Customer-Churn-Prediction)", unsafe_allow_html=True)

# Collapsible section for Project 6
with st.expander("Project 6: Telecommunication Customer Churn Prediction"):
    st.write("Short Description: Successfully conducted an in-depth analysis of a telecommunications company's customer churn dataset.")
    display_detailed_description_project_6()


# You can add more projects and sections as needed

# Add a separator
st.markdown("---")

# Contact Information
st.header("Contact Me")

# Create a container for center alignment
centered_container = st.container()

# Center align the "Contact Me" section
with centered_container:
    email = "Email: sakibimtiaz669@gmail.com"
    st.text(email)

    # Create a container for the social media links
    social_media_container = st.container()

    # Define the social media links
    social_media_links = [
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/search/results/all/?keywords=data%20entry&origin=GLOBAL_SEARCH_HEADER&sid=HRC",
            "icon": "https://raw.githubusercontent.com/devicons/devicon/master/icons/linkedin/linkedin-original.svg",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/sakib4535",
            "icon": "https://raw.githubusercontent.com/devicons/devicon/master/icons/github/github-original.svg",
        },
        {
            "name": "Kaggle",
            "url": "https://www.kaggle.com/sakibimtiaz",
            "icon": "https://www.vectorlogo.zone/logos/kaggle/kaggle-icon.svg",
        },
        {
            "name": "ResearchGate",
            "url": "https://www.researchgate.net/profile/Hasnain-Sakib",
            "icon": "https://www.researchgate.net/favicon.ico",
        },

    ]

    # Create three columns for the social media links
    col111111, col222222, col333333, col444444 = st.columns(4)

    # Display the social media links with icons in each column
    with col111111:
        for link in social_media_links[:1]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )

    with col222222:
        for link in social_media_links[1:2]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )

    with col333333:
        for link in social_media_links[2:3]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )

    with col444444:
        for link in social_media_links[3:]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )





