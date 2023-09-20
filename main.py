import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import json
import requests

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
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f0f0;
        margin: 0;
        padding: 0;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #002366 0%, #3366cc 100%);
        color: white;
    }

    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        text-align: center;
    }

    .profile-image {
        width: 100px;
        height: 100px;
        object-fit: cover;
        border-radius: 50%;
        margin-right: 20px;
    }

    .profile-image:hover {
        transform: scale(1.1);
        transition: transform 0.3s ease-in-out;
    }

    /* Add more custom styles as needed */
</style>

    """,
    unsafe_allow_html=True,
)

# Add a cover image
background_image = Image.open("background.gif")
st.image(background_image, use_column_width=True, caption="")

# Create a container for the header
header_container = st.container()

# Header
with header_container:
    # Profile picture on the left
    col1, col2 = st.columns([1, 4])
    with col1:
        profile_image = Image.open("F:/sakib.jpg")

        st.image(profile_image, width=210)


    # About Me section on the right
    with col2:
        st.markdown(
            "<h4 style='font-style: italic; font-weight: bold; font-family: cursive;'>In the depths of data, where patterns hide and insights slumber, I emerge as the seeker, deciphering the enigma of Machine Learning.</h4>",
            unsafe_allow_html=True)
        st.title("Hasnain Imtiaz")
        st.markdown("<i>Machine Learning, Data and Technology Enthusiast</i>", unsafe_allow_html=True)
        st.write(
            """ I possess a robust grasp of statistical concepts and command a strong understanding of cutting-edge 
            machine learning algorithms. My expertise extends to a wide array of methodologies, including but not limited 
            to Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), K-Nearest Neighbors (KNN),
             and Naive Bayes. I have effectively applied these algorithms in various independent projects, showcasing my 
             proficiency in harnessing their capabilities. In my pursuit of enhancing model performance and predictive 
             accuracy, I have skillfully employed advanced techniques such as ensembling methods and transfer learning. 
             Additionally, I have leveraged tools like TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings 
             to extract meaningful features from data, thereby augmenting the effectiveness of predictive models. 
             My commitment to excellence extends to the preprocessing stage, where I prioritize batch processing and 
             feature engineering during feature extraction. This meticulous approach ensures that the input data is 
             well-prepared to yield optimal results when subjected to machine learning algorithms. Furthermore, 
             I am driven by a strong interest in delving into Recurrent Neural Networks (RNN) and Transformer-based models. 
             These cutting-edge deep learning architectures hold significant promise for a wide range of natural language processing and sequence modeling tasks, and I am enthusiastic about exploring their potential applications in future projects.
 """
        )

# Load your Lottie JSON animation
    with open("animation_lmnn8dtf.json", "r") as json_file:  # Replace with the path to your Lottie JSON file
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
    st.header("Computational Skills")
    st.text("Here are some of the skills I've developed:")
    st.markdown("- Machine Learning")
    st.markdown("- Natural Language Processing")
    st.markdown("- Data Structure and Algorithms")
    st.markdown("- Data Manipulation")
    st.markdown("- Web Scraping")
    st.markdown("- Data Analysis")
    st.markdown("- Python Programming")
    st.markdown("- Database Interaction")
    st.markdown("- Data Manipulation")
    st.markdown("- Feature Engineering")
    st.markdown("- Exploratory Analysis")
    st.markdown("- Data Visualization")


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
            "name": "Kafka",
            "image_url": "https://www.vectorlogo.zone/logos/apache_kafka/apache_kafka-icon.svg",
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
            "name": "Ruby on Rails",
            "image_url": "https://raw.githubusercontent.com/devicons/devicon/master/icons/rails/rails-original-wordmark.svg",
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

# Center the Work Experience section
# Center the Work Experience section and underline it
st.markdown("<h1 style='text-align: center; text-decoration: underline;'>Work Experience</h1>", unsafe_allow_html=True)


# Create a container for the centered content
centered_container = st.container()

# Centered Work Experience
with centered_container:
    st.markdown("<h3 style='text-align: center;'>Student, Machine Learning Engineering</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Creative IT Institute, Professional IT Certificate</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>January 2023 - Present</h4>", unsafe_allow_html=True)
    # Style the description with a box
    st.markdown("""
          <div style='
              background-color: #f8f8f8;
              border: 2px solid #3366cc;
              border-radius: 10px;
              padding: 20px;
              text-align: left;
          '>
              <p style='font-size: 16px;'>
              This Professional Course dives into the theoretical and practical aspects of programming concepts in Python and 
              explores a wide range of topics in data science. We cover Python basics, object-oriented programming, and 
              advanced data structures. This course introduced different libraries: Pandas, Numpy, Matplotlib, and Seaborn, 
              Scikit-learn, OpenCV, PyTorch, for data manipulation, numerical computing, and visualization. Exploratory 
              Data Analysis (EDA) is done using real-world datasets like Iris and Haberman. Probability theory, including 
              Poisson and Gaussian distributions, is covered. NLP techniques like BOW, TF-IDF, and tokenization are introduced. 
              Machine Learning algorithms such as KNN, Naive Bayes, logistic regression, SVM, decision trees, and random forests 
              are explored. Clustering, sentiment analysis, recommendation systems, and cancer prediction projects are undertaken. 
              This course equips us with theoretical knowledge and practical skills in Python, data science, and machine learning.
              </p>
          </div>
      """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Research Assistant</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Microgovernance Research Initiatives</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>July 2021 - December 2021</h3>", unsafe_allow_html=True)
    st.markdown("""
             <div style='
                 background-color: #f8f8f8;
                 border: 2px solid #3366cc;
                 border-radius: 10px;
                 padding: 20px;
                 text-align: left;
             '>
                 <p style='font-size: 16px;'> During my tenure as a research assistant at MGR, a renowned public university, I had the opportunity to engage "
        "in a range of valuable experiences and training programs. One of the highlights was receiving training in advanced "
        "research techniques, which equipped me with the necessary skills to conduct high-quality research. Through workshops "
        "and seminars, I gained a deeper understanding of various research methodologies and data analysis techniques, "
        "allowing me to contribute effectively to research projects. Moreover, I had the privilege of participating in "
        "workshops conducted by prominent organizations such as the International Foundation for Electoral Systems (IFES)"
        " and BRAC. These workshops focused on important societal issues, with a particular emphasis on gender, society, "
        "misinformation, cyber threats, and social media platforms. Engaging in these workshops provided me with a "
        "comprehensive understanding of these topics and their impact on contemporary society. I gained insights into the "
        "challenges posed by gender inequality, the spread of misinformation in the digital age, the growing threat of "
        "cyber attacks, and the influence of social media platforms on public opinion and discourse." </p>
          </div>
      """, unsafe_allow_html=True)


# Projects
# Projects
st.header("Projects")
st.text("Here are some of the projects I've worked on:")

# Create a container for the project cards
project_container = st.container()

# Project 1: Game Data Analytics using Ratings, Playtime, Suggestions
with project_container:
    st.subheader("Game Data Analytics using Ratings, Playtime, Suggestions")

    # Create a single row to display the image and description side by side
    col1, col2 = st.columns([1, 2])  # Adjust the number of columns as needed

    # Display the image on the left
    with col1:
        st.image("relations-RF.png", width=400, caption="Sentiment Analysis Example")

    # Create a box for the project description and shift it to the right
    with col2:
        # Apply CSS styles to create a box and right shift it
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True,  # Allow HTML tags for styling
        )

        # Keywords section with separate black boxes and blue-colored text for each keyword
        keywords = [
            "Game Ratings",
            "Predictive Models",
            "Machine Learning",
            "RAWG API",
            "Decision Trees",
            "Random Forests",
            "Model Performance",
            "Feature Importance",
            "Regression Analysis",
        ]

        keyword_boxes = [
            f'<span style="background-color: #000; color: #3366cc; padding: 2px 5px; border-radius: 3px; margin-right: 2px;">{keyword}</span>'
            for keyword in keywords
        ]

        keywords_html = " ".join(keyword_boxes)

        # Keywords with right shift
        st.markdown(
            f"""
            <div style="text-align: right; margin-right: 20px;">  <!-- Adjust the margin for right shift -->
                <strong>Keywords:</strong> {keywords_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Project 2: Laptop Prices: Unveiling the Key Factors Influencing Pricing Patterns
with project_container:
    st.subheader("Laptop Prices: Unveiling the Key Factors Influencing Pricing Patterns")

    # Create a single row to display the left and right columns
    left_col, right_col = st.columns([1, 2])  # Adjust the column proportions as needed

    # Left column for graphs
    with left_col:
            st.image("project02.png", width=400, caption="Graph 1")
            st.image("laptop.png", width=400, caption="Graph 2")

    # Right column for project description and keywords
    with right_col:
        # Project description
        st.markdown(
            """
            <div style="
                border: 2px solid #3366cc;
                border-radius: 10px;
                padding: 20px;
                text-align: left;
                margin-left: 40px;  /* Adjust the margin for right shift */
            ">
            <ul>
            <li> Description: In today's fast-paced digital world, laptops have become an indispensable tool for 
              both work and leisure. With an overwhelming variety of options available in the market, understanding the factors 
              that determine laptop prices is crucial for making informed purchasing decisions. </li>
             <li> This project delves into the world 
              of laptop pricing, leveraging data analysis and machine learning techniques to uncover the key factors that influence 
              laptop prices.</li>
              <li> By analyzing a comprehensive dataset of laptop specifications and prices, this project aims to provide valuable 
              insights into the pricing patterns of laptops. Through the utilization of regression models, particularly Random 
              Forest Regression, we explore the relationships between various laptop features and their impact on pricing. 
              The project's findings empower consumers to make informed decisions based on their budget, performance requirements, 
              and desired features. </li>

              <li> Furthermore, feature importance analysis is conducted to identify the most influential factors driving laptop prices. 
              This analysis allows us to uncover which features have the greatest impact on pricing decisions. The results are 
              visualized through informative graphs, highlighting the relative importance of each feature and enabling users to
              prioritize their preferences based on budget and requirements. </li>
            """,
            unsafe_allow_html=True,  # Allow HTML tags for styling
        )

        # Keywords section with separate black boxes and blue-colored text for each keyword
        keywords = [
            "Laptop Prices",
            "Pricing Patterns",
            "Data Analysis",
            "Machine Learning",
            "Regression Models",
            "Feature Importance",
            "Budget",
            "Performance Requirements",
        ]

        keyword_boxes = [
            f'<span style="background-color: #000; color: #3366cc; padding: 2px 5px; border-radius: 3px; margin-right: 5px;">{keyword}</span>'
            for keyword in keywords
        ]

        keywords_html = " ".join(keyword_boxes)

        # Keywords with right shift
        st.markdown(
            f"""
            <div style="text-align: right; margin-right: 20px;">  <!-- Adjust the margin for right shift -->
                <strong>Keywords:</strong> {keywords_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Project 3:
with project_container:
    st.subheader("""Looking Through a National News Article Texts to learn about the name 'Joe Biden' !""")
    # Create a box for the project description
    st.markdown(
        """
        <div style="
            border: 2px solid #3366cc;
            border-radius: 10px;
            padding: 18px;
            text-align: left;
            margin-left: auto; /* Auto margin for center alignment */
            margin-right: auto; /* Auto margin for center alignment */
        ">
        Description: The aim of this project was to leverage Natural Language Processing (NLP) techniques to analyze a 
        large corpus of national news articles to track and understand the mentions of the name 'Joe Biden.' The project 
        incorporated various NLP methods, including Word Cloud generation, handling stop words, label encoding, and creating a
         Bag of Words representation.
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML tags for styling
    )

    # Create a single row to display the images horizontally
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed

    # Display the images horizontally within the same container
    with col1:
        st.image("g20.png", width=400, caption="Sentiment Analysis Example")
    with col2:
        st.image("g20-1.png", width=400, caption="Image Classification Example")
    with col3:
        st.image("g20-3.png", width=400, caption="Image Classification Example")

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




# Project 4: GamersHub: Text and Opinion Mining of Reddit Comments
with project_container:
    st.subheader("GamersHub: Text and Opinion Mining of Reddit Comments")


    # Create a box for the project description
    st.markdown(
        """
        <div style="
            border: 2px solid #3366cc;
            border-radius: 10px;
            padding: 20px;
            text-align: left;
            margin-left: auto; /* Auto margin for center alignment */
            margin-right: auto; /* Auto margin for center alignment */
        ">
        Description: GamersHub: Text and Opinion Mining of Reddit Comments
        In the initial stages, the dataset of Reddit comments is likely collected and preprocessed. This entails tasks such as data cleaning,
        removal of irrelevant symbols, and potentially splitting the data into training and testing sets using train_test_split from
        sklearn.model_selection. In this case, the CSV file contains 21,821 rows and 3 columns. Each row in the CSV file likely represents a
        separate record or data point, while each of the 3 columns likely holds different attributes, variables, or fields associated with the data.
        To convert the text data into a format suitable for analysis, the project employs the CountVectorizer and TfidfVectorizer from
        sklearn.feature_extraction.text. These techniques transform the comments into numerical vectors, enabling machine learning algorithms
        to process and analyze the text data effectively.
        For a deeper understanding of the underlying themes within the comments, the project utilizes LatentDirichletAllocation from
        sklearn.decomposition to perform topic modeling. This aids in uncovering latent topics or patterns within the discussions, contributing
        valuable insights into prevalent themes.
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML tags for styling
    )

    # Create a single row to display the images horizontally
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns as needed

    # Display the images horizontally within the same container
    with col1:
        st.image("reddit2.png", width=400, caption="Image 1")
    with col2:
        st.image("reddit1.png", width=400, caption="Image 2")
    with col3:
        st.image("reddit3.png", width=400, caption="Image 3")

    # Keywords section for Project 4
    keywords_project4 = [
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

    keyword_boxes_project4 = [
        f'<span style="background-color: #000; color: #3366cc; padding: 2px 3px; border-radius: 3px; margin-right: 5px;">{keyword}</span>'
        for keyword in keywords_project4
    ]

    keywords_html_project4 = " ".join(keyword_boxes_project4)

    # Center-aligned Keywords for Project 4
    st.markdown(
        f"""
        <div style="text-align: center;">
            <strong>Keywords:</strong> {keywords_html_project4}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    col1, col2, col3, col4 = st.columns(4)

    # Display the social media links with icons in each column
    with col1:
        for link in social_media_links[:1]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )

    with col2:
        for link in social_media_links[1:2]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )

    with col3:
        for link in social_media_links[2:3]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )

    with col4:
        for link in social_media_links[3:]:
            st.markdown(
                f'<a href="{link["url"]}" target="_blank" rel="noreferrer">'
                f'<img src="{link["icon"]}" alt="{link["name"]}" width="50" height="50"/>'
                f'</a>',
                unsafe_allow_html=True,
            )





