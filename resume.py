from llama_index.core import SummaryIndex, SimpleDirectoryReader, get_response_synthesizer

import re

from utils import config

config()


def get_resume_query(resume_path: str) -> str:
    reader = SimpleDirectoryReader(input_files=[resume_path])
    documents = reader.load_data(show_progress=True)

    index = SummaryIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    search_query = query_engine.query(
        """Create a search query for this resume general area of expertise only including letters and with a 5 words limit, prefer less words and try to be generic
        example: mid-level javascript software engineer
        """)

    clean = re.sub(' +', '+', str(search_query))

    return clean


resume = """
Johnathan M. Carter
1234 Maple Street, Apt 6B, Springfield, IL 62701
Phone: (555) 123-4567 | Email: john.carter@email.com
LinkedIn: linkedin.com/in/johnathanmcarter | GitHub: github.com/john-carter-dev

Professional Summary
Results-driven software engineer with 6+ years of experience in designing and developing scalable web applications and services. Adept in full-stack development, with a focus on optimizing application performance and user experiences. Proven track record of delivering complex projects on time while collaborating effectively across teams.

Work Experience
Senior Software Engineer
Techify Solutions – Chicago, IL | June 2021 – Present

Lead a team of 5 engineers to build a cloud-based platform for managing large-scale financial data, reducing data processing time by 30%.
Developed key microservices using Node.js, React, and AWS Lambda, improving system scalability.
Integrated third-party APIs, such as Stripe and Twilio, to enhance payment and communication features.
Collaborated with product managers to prioritize features, resulting in a 20% increase in user retention.
Software Engineer
ByteWorks – St. Louis, MO | July 2018 – May 2021

Engineered web applications using the MERN stack (MongoDB, Express, React, Node.js) for e-commerce platforms, reducing load times by 40%.
Implemented RESTful APIs and GraphQL endpoints to streamline client-server communication.
Automated testing using Jest and Cypress, improving code reliability and reducing bug reports by 15%.
Mentored junior developers, providing guidance on best practices in version control, testing, and code reviews.
Junior Software Developer
CodeCraft Innovations – Indianapolis, IN | August 2016 – June 2018

Contributed to the development of internal tools using Python and Django to automate business workflows.
Collaborated with designers to improve UI/UX, resulting in a 25% boost in user engagement.
Conducted code reviews and refactoring sessions to maintain high code quality standards.
Education
Master of Science in Computer Science
University of Illinois at Urbana-Champaign | 2016

GPA: 3.8/4.0
Focus: Distributed Systems, Data Structures, Cloud Computing
Bachelor of Science in Computer Engineering
Purdue University, West Lafayette | 2014

GPA: 3.7/4.0
Activities: ACM Programming Team, Robotics Club
Technical Skills
Languages: JavaScript, TypeScript, Python, Java, SQL
Web Technologies: React, Redux, Node.js, Express, GraphQL
Cloud Platforms: AWS (Lambda, S3, EC2), Google Cloud
Databases: MongoDB, PostgreSQL, MySQL
Tools: Docker, Kubernetes, Jenkins, Git, Terraform
Testing Frameworks: Jest, Mocha, Cypress, Selenium
Certifications
AWS Certified Solutions Architect – Associate | 2020
Google Cloud Professional Cloud Architect | 2021
Projects
Expense Tracker App – Full-Stack Application

Developed a personal finance tracker using React, Node.js, and MongoDB, allowing users to log expenses and generate reports.
Deployed the application on AWS using Docker and Kubernetes.
Real-Time Chat Application – WebSocket-based Messaging App

Built a real-time chat application using React and WebSockets, featuring group chats, message notifications, and file sharing.
Integrated Redis for session management and message queueing.
Interests & Volunteer Work
Passionate about open-source contributions, with regular participation in Hacktoberfest and maintaining a small collection of npm packages.
Volunteer mentor at Code for Good, helping underrepresented students learn programming and computer science basics.
"""
