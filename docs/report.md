# 1. Timely Trends of Job Postings
- **Project Title:** Timely Trends of Job Postings
- **Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang**
- **Author Name:** Akshaya Narayana
- **GitHub:** https://github.com/AkshayaNarayana
- **LinkedIn:** https://www.linkedin.com/in/akshaya-narayana-8221201ab
- **PowerPoint Presentation:** https://github.com/AkshayaNarayana/UMBC-DATA606-Capstone/blob/main/docs/Capstone_Akshaya.pptx
- **Youtube Video:** https://youtu.be/oOgapn8aQTI

# 2. Background
### What is it about?

The dataset includes job posting details such as job ID, experience requirements, qualifications, salary range, location, company information, and more. It provides a complete overview of career prospects in a variety of industries, locations, and firms. It is a valuable resource for conducting employment market dynamics research and may be used for activities such as trend identification, geographic evaluation, and natural language processing (NLP) investigation of job descriptions.

### Why does it matter?

Anyone involved in the hiring process, from recruiters to lawmakers, has a vested interest in comprehending the data about job postings. People looking for jobs can learn more about the skills and abilities that employers want, and recruiters can look at patterns in how they hire people. Policymakers can use this kind of data to learn about the job market and make smart choices.

### What are your research questions?

1. How has the volume of job postings changed over the years?
2. Is there any seasonality or periodicity in job postings throughout the year?
3. What are the top skills consistently in demand across different roles and industries?
4. How do skill requirements vary among different job roles and industries?

# 3. Dataset
- **Data source:** https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset/data

- **Data size:** 1.74 GB

- **Data shape:** The dataset contains 1615940 rows and 23 columns.

- **Time period:** The data covers job postings from September 2021 to September 23.

- **What does each row represent?**

  Each row represents a unique job posting.
  
- **Data dictionary**
  ##### 1. Job Id 
  - Data type: int64 
  - Definition: A unique identifier for each job posting.
   
  #### 2. Experience:
  - Data Type: object 
  - Definition: The required or preferred years of experience for the job.
  - Potential Values: Various experience ranges (e.g., "5 to 8 Years").
    
  #### 3. Qualifications:   
  - Data Type: object 
  - Definition: The educational qualifications needed for the job.
  - Potential Values: Different educational qualifications (e.g., "BBA", "BA").

  #### 4. Salary Range:
  - Data Type: object 
  - Definition: The range of salaries or compensation offered for the position.
  - Potential Values: Various salary ranges (e.g., "$59K-$88K").
  
  #### 5. Location:
  - Data Type: object 
  - Definition: The city or area where the job is located.
  - Potential Values: Different city or area names.

  #### 6. Country:
  - Data Type: object 
  - Definition: The country where the job is located.
  - Potential Values: Different country names.

  #### 7. Latitude:
  - Data Type: float64
  - Definition: The latitude coordinate of the job location.

  #### 8. Longitude:
  - Data Type: float64
  - Definition: The longitude coordinate of the job location.

  #### 9. Work Type:
  - Data Type: object
  - Definition: The type of employment (e.g., full-time, part-time, contract).
  - Potential Values: Various work types (e.g., "Part-Time", "Temporary").
    
  #### 10. Company Size:
  - Data Type: int64
  - Definition: The approximate size or scale of the hiring company.

  #### 11. Job Posting Date:
  - Data Type: object 
  - Definition: The date when the job posting was made public.

  #### 12. Preference:
  - Data Type: object 
  - Definition: Special preferences or requirements for applicants.

  #### 13. Contact Person:  
  - Data Type: object 
  - Definition: The name of the contact person or recruiter for the job.

  #### 14. Contact:  
  - Data Type: object 
  - Definition: Contact information for job inquiries.

  #### 15. Job Title:  
  - Data Type: object 
  - Definition: The job title or position being advertised.

  #### 16. Role:  
  - Data Type: object 
  - Definition: The role or category of the job.

  #### 17. Job Portal:  
  - Data Type: object 
  - Definition: The platform or website where the job was posted.

  #### 18. Job Description:  
  - Data Type: object 
  - Definition: A detailed description of the job responsibilities and requirements.

  #### 19. Benefits: 
  - Data Type: object 
  - Definition: Information about benefits offered with the job.

  #### 20. Skills:  
  - Data Type: object 
  - Definition: The skills or qualifications required for the job.

  #### 21. Responsibilities:  
  - Data Type: object 
  - Definition: Specific responsibilities and duties associated with the job.

  #### 22. Company:  
  - Data Type: object 
  - Definition: The name of the hiring company.

  #### 23. Company Profile: 
  - Data Type: object 
  - Definition: A brief overview of the company's background and mission.
 
    
- **Which variable/column will be your target/label in your ML model?**
  
  The Number of Job Openings will be the target for timely trends.

  
- **Which variables/columns may be selected as features/predictors for your ML models?**

  Factors like the date when the job posting was made could capture temporal patterns and seasonality in job openings.

# 4. Exploratory Data Analysis (EDA)
- Imported the dataset and examined it for null values, ensuring data completeness and accuracy.
- Streamlined the dataset by removing redundant fields like 'Job Id', 'Preference', 'Contact Person', 'Contact', 'Job Portal',' Benefits', 'Company Profile', etc., optimizing it for further analysis.
- Transformed the 'Job Posting Date' column into a datetime format to facilitate time-based analysis.
- Utilized Matplotlib and Seaborn to generate visual representations to analyze the distribution and patterns of job vacancies over some time. This involved the use of bar plots to represent annual job opportunities and line plots to represent monthly job vacancies.
- Calculated the number of roles for each date by grouping the data according to the 'Job Posting Date' and 'Role', and then combined the counts to obtain the total number of job openings per date.

# 5. Model Training
- For predictive analytics, we employed the ARIMA model and the auto ARIMA model to forecast job openings. ARIMA, which stands for AutoRegressive Integrated Moving Average, is well-suited for time series forecasting, especially for datasets with trends and seasonality.
- Split the data into training and testing sets, using the first part for training and the last 30 entries for testing, allowing validation of unseen data.
- Conducted the Augmented Dickey-Fuller (ADF) test to check for stationarity in the time series. The initial p-value was 0.899, indicating non-stationarity. Then differenced the data to make it stationary, achieving a p-value of approximately 1.7296×10⁻²⁰, indicating stationarity after differencing.
- Plotted the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF) to determine the orders for training the ARIMA model.
- Used the trained ARIMA model to forecast the test data and visualize the results.
- Saved the trained ARIMA model using the pickle library to facilitate future predictions without retraining.
- Python Packages used:
  - pandas for data manipulation and analysis.
  - statsmodels for time series analysis and ARIMA model implementation
  - pmdarima library's auto_arima function to automatically select the best ARIMA model parameters.
  - matplotlib for plotting data and diagnostics.
  - pickle for saving and loading trained models. 
- Development and model training were carried out on a local machine using Jupyter Notebooks and on Google Colab for leveraging computational resources.

# 6. Application of the Trained Models
Developed a web application that allows users to interact with the trained model. 
- Streamlit: <img src="" style="display: block; margin-left: auto; margin-right: auto; width: 400px; height: 200px;">
  
# 7. Conclusion
- The project focused on analyzing job posting data to forecast future job openings using ARIMA and auto ARIMA models. Through rigorous data preprocessing, model selection, and validation, we developed predictive models that can assist in workforce planning. The models were validated on unseen data and saved for future use.
- Limitations such as the reliance on historical data patterns and potential data quality issues, this work provides a valuable tool for anticipating labor market trends.
- Lessons learned emphasize the importance of data preprocessing, model selection, and validation in time series analysis. Continuous model refinement and evaluation are crucial for maintaining accurate predictions.
- Future improvements could involve integrating external factors and exploring advanced modeling techniques to enhance predictive accuracy and robustness.
