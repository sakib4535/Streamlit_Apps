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
st.image(background_image, use_container_width=True, caption="")

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
            """ I possess a deep and methodical understanding of statistical principles, enabling me to command robust expertise in machine learning, spanning supervised algorithms like 
            regression and classification, unsupervised techniques such as clustering and dimensionality reduction, and 
            reinforcement learning for dynamic decision-making and optimization, all complemented by a strong emphasis on 
            hyperparameter tuning, model adaptation, and solving complex, domain-specific challenges.
             Throughout my academic journey and professional endeavors, I have designed, developed, and deployed 
             sophisticated machine learning and deep learning algorithms. This includes applying neural networks, 
             bagging and boosting techniques (such as XGBoost, LightGBM, and AdaBoost), and graph-based algorithms 
             to tackle diverse challenges like text analysis and mining, anomaly detection, emotion recognition, and social network analysis. 
             Leveraging principles of information retrieval and extraction, I have conducted in-depth text mining 
             on unstructured data to derive actionable insights. 
             I am deeply committed to fostering interdisciplinary collaborations at the nexus of computational science, 
             mathematics, social science, and human communication. By integrating diverse methodologies, I aim to create 
             channels that process information through multiple dimensions, ensuring nuanced and realistic interpretations
              of complex phenomena. My approach emphasizes filtering out noise and enhancing signal accuracy, enabling 
              robust predictions and innovative solutions. This philosophy is rooted in the belief that multidimensional 
              perspectives enrich our understanding and pave the way for transformative discoveries.
 """
        )



import streamlit as st

import streamlit as st

# Title with double underline
st.markdown("""
    <h1 style="
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        text-align: center;
        color: #2C3E50;
        margin-bottom: 30px;
        position: relative;
    ">
        Core Interests and Competencies
        <div style="
            position: absolute;
            bottom: -5px;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(to right, #3498db, #9b59b6, #f39c12);
            border-radius: 5px;
        "></div>
        <div style="
            position: absolute;
            bottom: -10px;
            left: 0;
            right: 0;
            height: 2px;
            background: #2C3E50;
            border-radius: 5px;
        "></div>
    </h1>
""", unsafe_allow_html=True)

# Define your interest zones and keywords
interest_zones = {
    "Communication Data Science": [
        "📊 Social Media Analytics",
        "💬 Text/Opinion Mining",
        "📰 Media Content Analysis",
        "🔍 Segmentation Analytics",
        "🌐 Network Analysis",
        "🧠 Sentiment Analysis",
        "📊 Anomaly Detection",
        "🗂️ Hate Speech Detection",
        "💬 Political Affiliation Detection"
    ],
    "Natural Language Processing": [
        "📝 Word Embedding", "🔢 TF-IDF", "🔠 Vectorization", "🗣️ Bag of Words",
        "🏷️ Name Entity Recognition (NER)", "📉 Stemming/Lemmatization",
        "🧩 Syntactic Parsing", "🔗 Dependency Parsing", "🔍 Topic Modeling",
        "📑 Document-Term Matrix", "📈 Text Classification", "⚖️ Performance Metrics", "🛠️ Seq2Seq", "🔍Attention", "📚 LLM (Large Language Model)"
    ],
    "Machine and Deep Learning": [
        "🔄 Ensemble Methods (LightGBM, XGBoost, AdaBoost, GBM)",
        "🤖 Neural Networks (CNN, RNN, LSTM)", "🔄 Transfer Learning",
        "🏷️ Sequence Labeling", "🔍 Attention Mechanisms", "📚 Supervised Learning",
        "🔄 Unsupervised Learning (Clustering)", "🔧 Feature Engineering",
        "🚫 Overfitting and Regularization", "🔍 Model Evaluation and Validation"
    ],
    "Data Visualization": [
        "📊 Data Insights", "📉 Interactive Charts", "📖 Data Storytelling", "📱 Data Manipulation and Scaling", "🌍 Geospatial Data Visualization", "📊 Interactive Dash Board"
    ]
}

# Creating four columns for each interest zone
col1, col2, col3, col4 = st.columns(4)

# Custom HTML for section titles (single underline)
def create_title_with_underline(title):
    return f"""
    <h2 style='
        font-family: "Arial", sans-serif; 
        font-size: 26px; 
        text-align: center; 
        color: #2C3E50; 
        margin-bottom: 20px;
        position: relative;
    '>
        {title}
        <div style='
            position: absolute;
            bottom: -5px;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(to right, #3498db, #9b59b6, #f39c12);
            border-radius: 5px;
        '></div>
    </h2>
    """

# Display each interest zone in its respective column with styled titles
with col1:
    st.markdown(create_title_with_underline("Communication Data Science"), unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{keyword}</li>" for keyword in interest_zones["Communication Data Science"]]) + "</ul>", unsafe_allow_html=True)

with col2:
    st.markdown(create_title_with_underline("Natural Language Processing"), unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{keyword}</li>" for keyword in interest_zones["Natural Language Processing"]]) + "</ul>", unsafe_allow_html=True)

with col3:
    st.markdown(create_title_with_underline("Machine and Deep Learning"), unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{keyword}</li>" for keyword in interest_zones["Machine and Deep Learning"]]) + "</ul>", unsafe_allow_html=True)

with col4:
    st.markdown(create_title_with_underline("Data Visualization"), unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{keyword}</li>" for keyword in interest_zones["Data Visualization"]]) + "</ul>", unsafe_allow_html=True)

# CSS styling for horizontal row layout
st.markdown("""
    <style>
        .grid-container {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            gap: 10px;
            flex-wrap: wrap;
        }
        .grid-box {
            flex: 1 1 23%;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            min-width: 200px;
            max-width: 23%;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .keyword-list {
            list-style-type: disc;
            margin-left: 20px;
            font-size: 16px;
            color: #444;
        }
    </style>
""", unsafe_allow_html=True)

#####################
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



######################################################################
# Sample data for demonstration
projects = [
    {"image": "drug.png", "description": ""},
    {"image": "boxplot_total_votes.png", "description": ""},
    {"image": "data_science.png", "description": ""},
    # {"image": "eth.png", "description": "Financial reporting in Excel."},
    {"image": "netflix.png", "description": ""},
    {"image": "nvidia.png", "description": ""},
    {"image": "disney.png", "description": ""},
    # {"image": "path/to/image8.jpg", "description": "Employee performance dashboard in Excel."},
    # {"image": "path/to/image9.jpg", "description": "Year-over-year growth analysis in Tableau."}
]

# Section header
st.title("Data Visualization - From My Projects")
st.write("Showcasing some of my best data visualizations and insights using Excel and Tableau.")

# Display the projects
for i in range(0, len(projects), 3):  # Loop through projects in chunks of 3
    cols = st.columns(3)  # Create 3 columns for the current row
    for j in range(3):  # Loop through each column in the row
        if i + j < len(projects):  # Ensure the project index is within bounds
            project = projects[i + j]
            with cols[j]:  # Place content in the current column
                # Display the image
                image = Image.open(project["image"])  # Load the image
                st.image(image, use_container_width=True)
                # Display the description
                st.write(f"**{project['description']}**")


################################################

###############################################

# Sample video data
videos = [
    {
        "title": "🤖Voice Assistant - Falcon LLM🔊",
        "video_path": "text-to-voice.mp4",
        "description": "A voice assistant powered by Falcon LLM, designed to understand and respond intelligently to user commands"
    },
    {
        "title": "📲Learning Information With API Interaction - InfoCare for USA🛰️",
        "video_path": "my_app.mp4",
        "description": "An API-based platform that provides real-time information and services for users in the USA, focusing on personalized learning and assistance."
    },
    {
        "title": "💻Searching For News - Real-Time Interaction📡",
        "video_path": "News_App.mp4",
        "description": "A real-time news search system that allows users to interact and retrieve the latest news with instant updates and responses."

    },
    {
        "title": "🧑‍Mapping Emotions of Texts (NLP Driven) - EmotionMapping 💻",
        "video_path": "Project.mp4",
        "description": "A natural language processing tool that analyzes text and maps emotional sentiment for insights into user reactions and feelings."
    }
]

# Section header
st.title("Project Videos")
st.write("Watch the project demonstrations showcasing the practical implementations of my work.")

# Display videos in a grid (2 columns per row for better visibility)
for i in range(0, len(videos), 2):
    cols = st.columns(2)  # Create a row with 2 columns
    for j in range(2):
        if i + j < len(videos):  # Ensure video exists
            video = videos[i + j]
            with cols[j]:
                # Display video title
                st.subheader(video["title"])
                # Display video
                st.video(video["video_path"])
                # Display description
                st.write(video["description"])



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

    # Display images in a horizontal row using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("relations-RF.png", width=400, caption="Random Forest Tree Algorithm (RF)")
    with col2:
        st.image("DTvsRF.png", width=400, caption="Difference Between RF and DF Algorithms")
    with col3:
        st.image("relations-DT.png", width=400, caption="Decision Tree Algorithm (DF)")

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/your-username/your-repository)", unsafe_allow_html=True)

# Collapsible section for the project
with st.expander("Project 1: Game Data Analytics using Ratings, Playtime, Suggestions"):
    display_detailed_description()



# Function to display detailed description of Project 2
def display_detailed_description_project_2():
    st.subheader("***Laptop Prices: Unveiling the Key Factors Influencing Pricing Patterns***")

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
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display images in a horizontal row using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("laptop.png", width=400, caption="Graph 1")
    with col2:
        st.image("project02.png", width=400, caption="Graph 2")
    with col3:
        st.image("laptop3.png", width=400, caption="Graph 2")

    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/sakib4535/Projects-ML-Analytics/tree/main/Laptop%20Price%20Analysis)", unsafe_allow_html=True)

# Collapsible section for Project 2
with st.expander("Project 2: Laptop Prices: Unveiling the Key Factors Influencing Pricing Patterns"):
    display_detailed_description_project_2()



# Function to display detailed description of Project 3
def display_detailed_description_project_3():
    st.subheader("""***API Security: Access Behavior and Anomaly Detection With Boosting Algorithms (AdaBoost, Gradient Boost and XGBoost)***""")

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
    Description: Detecting unusual API behaviors is essential for cybersecurity, 
    particularly in sectors handling sensitive data. The dataset serves as a window into the intricate dynamics 
    between microservice-based applications and their gateway to the digital world—Application Programming Interfaces
     (APIs). In this realm, APIs act as the linchpin, facilitating seamless interactions between applications and 
     programmatic functions.. High accuracy (83-86%) and strong ROC-AUC scores reflect reliable anomaly detection, 
     while high precision and recall ensure efficient threat identification with minimal false alarms. 
     For example, a high recall rate shows that the models consistently capture actual anomalies—an essential 
     quality in environments with critical data.
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display images in a horizontal row using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("cy1.png", width=400, caption="Sentiment Analysis Example")
    with col2:
        st.image("cy2.png", width=400, caption="Image Classification Example")
    with col3:
        st.image("cy3.png", width=400, caption="Image Classification Example")

    # Keywords section for Project 3
    keywords_project3 = [
        "API Security",
        "Vulnerabilities",
        "Anomaly Detection",
        "Suspicious API ",
        "AdaBoost",
        "Gradient Boost",
        "XGBoost",
        "Security Threats",
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
    st.markdown("[GitHub Repository](https://github.com/sakib4535/Cyber-Security-Data-Research)", unsafe_allow_html=True)

# Collapsible section for Project 3
with st.expander("Project 3: API Security: Access Behavior and Anomaly Detection With Boosting Algorithms"):
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
    Description:

    In the initial stages, the dataset of Reddit comments is likely collected and preprocessed. This entails tasks such as data cleaning,
    removal of irrelevant symbols, and potentially splitting the data into training and testing sets using `train_test_split` from
    `sklearn.model_selection`. In this case, the CSV file contains 21,821 rows and 3 columns. Each row in the CSV file likely represents a
    separate record or data point, while each of the 3 columns likely holds different attributes, variables, or fields associated with the data.

    To convert the text data into a format suitable for analysis, the project employs the `CountVectorizer` and `TfidfVectorizer` from
    `sklearn.feature_extraction.text`. These techniques transform the comments into numerical vectors, enabling machine learning algorithms
    to process and analyze the text data effectively.For a deeper understanding of the underlying themes within the comments, the project utilizes `LatentDirichletAllocation` from
    `sklearn.decomposition` to perform topic modeling. This aids in uncovering latent topics or patterns within the discussions, contributing
    valuable insights into prevalent themes.
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display the images in columns
    st.subheader("Images:")
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed
    with col1:
        st.image("reddit2.png", width=350, caption="Image 1")
    with col2:
        st.image("reddit1.png", width=350, caption="Image 2")
    with col3:
        st.image("reddit3.png", width=350, caption="Image 3")

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
    st.subheader("***Multi-Label Text Sentiment Classification and Word Embedding Analysis for the Review of 'Thread' Social Media Application***")

    # Description content inside a markdown container
    description_content = """
    <div style="
        border: 2px solid #3366cc;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        margin-left: 40px; /* Right shift for alignment */
    ">
    📊 Comprehensive dataset of over 37,000 reviews capturing extensive user feedback from both Android and iOS platforms, 
    offering insights into the "Thread" app's reception. 📝

    💬 Achieved label-wise accuracies: Positive Sentiment at 88.25%, Negative Sentiment at 98.20%, and Neutral Sentiment at 91.00%. 
    With a mean accuracy of 92.48%, the model demonstrates strong performance across sentiment categories. 📈

    🔍 Leveraged word embedding analysis for detailed text classification, identifying prevalent themes within user sentiments. 🌐

    🤖 Applied multi-label text sentiment classification methods, integrating diverse data to enhance understanding of user perspectives. 🧠

    🔮 Insights from this project contribute to understanding social media feedback dynamics, aiding in app feature improvements and customer satisfaction strategies. 💼
    </div>
    """
    st.markdown(description_content, unsafe_allow_html=True)  # Display the HTML content in Markdown

    # Display the images in columns
    st.subheader("Images:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("th1.png", width=350, caption="Image 1")
    with col2:
        st.image("th2.png", width=350, caption="Image 2")
    with col3:
        st.image("th3.png", width=350, caption="Image 3")

    col4, col5, col6 = st.columns(3)
    with col4:

        st.image("th4.png", width=850, caption="Image 4")


    # GitHub Link
    st.subheader("GitHub Link:")
    st.markdown("[GitHub Repository](https://github.com/sakib4535/Text-Mining-and-Opinion-Reddit/tree/main)", unsafe_allow_html=True)

# Collapsible section for Project 5
with st.expander("Project 5: Multi-Label Text Sentiment Classification and Word Embedding Analysis for the Review of 'Thread' Social Media Application"):
    st.write("Short Description: Multi-label sentiment analysis and word embedding analysis for a social media application.")
    display_detailed_description_project_5()





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

    st.subheader("Images:")
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed
    with col1:
        st.image("tel1.png", width=350, caption="Image 1")
    with col2:
        st.image("tel2.png", width=350, caption="Image 2")
    with col3:
        st.image("tel3.png", width=350, caption="Image 3")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image("tel4.png", width=350, caption="Image 4")
    with col5:
        st.image("tel5.png", width=350, caption="Image 5")
    with col6:
        st.image("tel6.png", width=350, caption="Image 6")

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





