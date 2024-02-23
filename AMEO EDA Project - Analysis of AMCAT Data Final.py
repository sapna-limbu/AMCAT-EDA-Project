#!/usr/bin/env python
# coding: utf-8

# # AMCAT Data Analysis
# - AMCAT (Aspiring Minds Computer Adaptive Test) is an employability assessment test used by companies to evaluate the job-readiness of candidates. The test assesses various skills such as aptitude, technical knowledge, and communication skills.

# ## Problem Statement :
# - Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.”
# - Test this claim with the data given to you.
# -  Is there a relationship between gender and specialization? (i.e. Does the preference of Specialization depend on the Gender?)
# 

# ##  Objective :
# 
# - Explore and analyze the dataset to determine if there is a relationship between gender and specialization? (i.e. Does the preference of Specialization depend on the Gender?)

# In[1]:


# Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import datetime as dt


# In[2]:


# Loading the dataset
df = pd.read_csv("data.xlsx - Sheet1.csv")


# In[3]:


## checking the dimensions of dataset
df.shape


# In[4]:


df.head()


# In[5]:


#There is an unnamed column and we cannot use this for data analysis. So we need to drop that column. 
df.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[6]:


df.head(5)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[ ]:





# In[9]:


df.isnull().sum()


# ## Dataset consists of
# - 27 Numerical columns
# - 9 catogorical columns
# - 2 datetime

# In[ ]:





# ## Data Cleaning

# In[10]:


### finding the anomolities in data
for i in df.columns:
    print('*'*20,i,'*'*20)
    print(df[i].unique())


# ## Observations of the columns(regarding cleaning and missing values):
# - 1.We can see the DOJ,DOL,DOB are given in timestamp format
# - 2.Job city column contains -1 values which are NaN equivalents.
# - 3.10 board column contain 0 value which is missing value
# - 4.12 board column contain 0 value which is missing value
# - 5.college state column contain 'union teritory' which is not a specific state
# - 6.Graduation year column contain 0 which is a missing value
# - 7.Domain column contain -1 which is a missing value
# 

#  

# In[11]:


### We can see the DOJ and DOL are given in timestamp format. Converting timestamp into date using datetime module.
## In DOL column,We can see the value 'present'. We will convert this into the present date for our analysis.

df["DOJ"]=pd.to_datetime(df["DOJ"]).dt.date
df["DOL"].replace("present",dt.datetime.today(),inplace=True)
df['DOL'] = pd.to_datetime(df['DOL']).dt.date

## Converting feature from DOJ and DOL as we are only concerned with how many years the person has worked in the organisation.
df['Experience'] = pd.to_datetime(df["DOL"]).dt.year - pd.to_datetime(df['DOJ']).dt.year

## Converting DOB column from timestamp to year
df['DOB'] = pd.to_datetime(df['DOB']).dt.year


# In[12]:


df.head(5)


# In[13]:


(df==0).astype(int).sum(axis=0)


# In[14]:


df.isin([-1, 'NaN']).sum()


# In[15]:


branch=['ComputerProgramming','ElectronicsAndSemicon','ComputerScience','MechanicalEngg','ElectricalEngg','TelecomEngg','CivilEngg']


# In[16]:


for i in branch:
    df[i]=df[i].replace(-1,np.nan)


# ## Dropping columns

# In[17]:


df.drop(columns=['CollegeID','CollegeCityID','CollegeCityTier'],axis=1,inplace=True)


# In[18]:


df.describe(include='object')


# In[19]:


df.describe(include='number')


# In[20]:


col = list(df.select_dtypes(include='number').columns)


# In[21]:


col


# In[22]:


out_dict={}
for i in col:
    Q1 = df[i].quantile(0.05)
    Q3 = df[i].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)]
    out_dict[i]=outliers


# In[23]:


len_out={}
for i in col:
    Q1 = df[i].quantile(0.05)
    Q3 = df[i].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)]
    len_out[i]=len(outliers)


# In[24]:


col1 = list(df.select_dtypes(include='object').columns)


# In[25]:


col1


# In[ ]:





# ## Column: '10-board'
# This column has missing values in the form of '0'.

# In[26]:


df['10board'].value_counts()


# In[27]:


df['10board']=df['10board'].replace('0','N/A')


# In[28]:


df['10board'].value_counts()


# In[29]:


board10=list(df['10board'].unique())


# In[30]:


state_10=[]
cbse_10=[]
icse_10=[]
for i in board10:
    if i in ('cbse','cbse[gulf_zone]','cbse ','cbsc','new delhi','board of secondary education'):
        cbse_10.append(i)
    elif i in ('icse','icse board','cicse'):
        icse_10.append(i)
    else:
        state_10.append(i)


# In[31]:


for i in state_10:
    df['10board'].replace(i,'State',inplace=True)
for i in cbse_10:
    df['10board'].replace(i,'CBSE',inplace=True)
for i in icse_10:
    df['10board'].replace(i,'ICSE',inplace=True)


# In[32]:


df['10board'].value_counts()


# ## Column: '12-board'
# This column has missing values in the form of '0'.

# In[33]:


df['12board'].value_counts()


# In[34]:


df['12board']=df['12board'].replace('0','N/A')


# In[35]:


board12=list(df['12board'].unique())


# In[36]:


state_12=[]
cbse_12=[]
icse_12=[]
for i in board12:
    if i in ('cbse','cbese ','cbsc','new delhi','cbse board','bice'):
        cbse_12.append(i)
    elif i in ('icse','ise board','cicse','isce','isc'):
        icse_12.append(i)
    else:
        state_12.append(i)


# In[37]:


for i in state_12:
    df['12board'].replace(i,'State',inplace=True)
for i in cbse_12:
    df['12board'].replace(i,'CBSE',inplace=True)
for i in icse_12:
    df['12board'].replace(i,'ICSE',inplace=True)


# In[38]:


df['12board'].value_counts()


# ## Column : 'Designation'
# Designation Column has 'get' value which is a not a desired value.We should clean this and can be imputed with mode of the column.

# In[39]:


df["Designation"].value_counts()


# In[40]:


df["Designation"].unique()


# In[41]:


# Filter out rows where "Designation" is 'get'
get_designations = df[df["Designation"] == 'get']

# Print the filtered DataFrame
print(get_designations)


# In[42]:


df[df["Designation"]=="get"][['Designation','JobCity','Salary','Specialization']]


# ### We can see that most of people whose Designation is unknown are from mechanical domain(70%) and ECE(30%).
# ### So we can pick the mode of designation for people belonging to mechanical and  electrical domain. And impute it with get value.

# In[43]:


#for people with mechanical engineering
Mech = df[df['Specialization'].isin(['mechanical engineering','mechanical and automation'])]['Designation'].mode()[0]

#for people with electronics and electrical engineering
EEE = df[df['Specialization']==('electronics and electrical engineering')]['Designation'].mode()[0]

print(f'mode for mechanical:  {Mech}\nmode for EEE:  {EEE}')


# In[44]:


# Now we will impute the 'get' of designation with modes of the respective domains

#For mechanical domain
df.loc[df['Specialization'].isin(['mechanical engineering','mechanical and automation']),'Designation'].replace('get',Mech,inplace=True)

#for EEE domain
df['Designation'].replace('get',EEE,inplace=True)


# In[ ]:





# ## Column : 'Jobcity'
# Jobcity contains missing values(-1). We need to treat this by using mode.

# In[45]:


df['JobCity']


# In[46]:


get_ipython().system('pip install  fuzzywuzzy')


# In[47]:


from fuzzywuzzy import process

def correct_spelling_errors(target_word="", choices=[], threshold=80):
    match, score = process.extractOne(target_word, choices)
    if score >= threshold:
        return match
    else:
        return target_word


# In[48]:


choices = ["Bangalore","Indore","Chennai","Gurgaon","Hyderabad","Kolkata","Pune","Noida","Mohali","Jhansi","Delhi","Bhubaneswar",
           "Mumbai","Mangalore","Rewari","Gaziabad","Bhiwadi","Mysore","Rajkot","Jaipur","Thane","Maharajganj","Thiruvananthapuram",
           "Punchkula","Coimbatore","Dhanbad","Lucknow","Gandhi Nagar","Unnao","Daman and Diu","Visakhapatnam","Nagpur","Bhagalpur",
           "Jaisalmer","Ahmedabad","Kochi/Cochin","Bankura","Kanpur","Vijayawada","Beawar","Alwar","Siliguri","Raipur","Bhopal",
           "Faridabad","Jodhpur","Udaipur","Muzaffarpur","Bulandshahar","Haridwar","Raigarh","Aurangabad","Belgaum","Dehradun",
           "Rudrapur","Jamshedpur","Dharamshala","Hissar","Ranchi","Chandigarh","Australia","Cheyyar","sonepat","Pantnagar","Jagdalpur",
           "Angul","Karad","Rajpura","Pilani","Ambala City","Gorakhpur","Patiala","Sambalpur","Haldia","Karnal","Vellore","Dausa",
           "Rourkela","Guwahati","Mohali","Phagwara","Baripada","Meerut","Yamuna Nagar","Shahibabad","Pondichery","Ras Al Khaimah",
           "Jalandhar","Manesar","vapi","Allahabad","Khopoli","Keral","Howrah","Patna","Nellore","Varanasi","Kakinada","Rayagada",
           "Bahadurgarh","Kota","Bhilai","Kolhapur","Surat","Durgapur","Mettur","Nagari","Johannesburg","Bathinda","Joshimath","Kharagpur",
           "London","Kurnool","Tirupati","Bhopal","Jeddah","Nalagarh","Jhajjar","Gulbarga","Muvattupuzha","Shimla","'Bilaspur",
           "Chandrapur","Nanded","Dharmapuri","Vandavasi","Rohtak","Asansol","Tirunelvelli","Ernakulam","Baroda","Ariyalur","Jowai",
           "Neemrana","Dubai","Ahmednagar","Nashik","Bellary","Ludhiana","Gagret","Indirapuram","Gwalior","Hospete","Miryalaguda",
           "Ganjam","Dharuhera","Hubli","Agra","kudankulam","Ongole","Bikaner","Jammu","Al Jubail","Kalmar","Sweden","Jaspur","Burdwan",
           "Shahdol","NCR-Delhi","Vizag"]


# In[49]:


df['JobCity'] = df['JobCity'].apply(lambda city: correct_spelling_errors(str(city), choices))


# In[50]:


df['JobCity'].nunique()


# In[51]:


df["JobCity"].replace("Bengaluru","Bangalore",inplace=True)


# In[54]:


df['JobCity']=df['JobCity'].replace('-1','N/A')


# In[55]:


df['JobCity'].value_counts()


# ## Specialization column

# In[56]:


df['Specialization'].unique()


# In[57]:


df['Specialization'].value_counts()


# In[58]:


specialization_map = \
{'electronics and communication engineering' : 'EC',
 'computer science & engineering' : 'CS',
 'information technology' : 'CS' ,
 'computer engineering' : 'CS',
 'computer application' : 'CS',
 'mechanical engineering' : 'ME',
 'electronics and electrical engineering' : 'EC',
 'electronics & telecommunications' : 'EC',
 'electrical engineering' : 'EL',
 'electronics & instrumentation eng' : 'EC',
 'civil engineering' : 'CE',
 'electronics and instrumentation engineering' : 'EC',
 'information science engineering' : 'CS',
 'instrumentation and control engineering' : 'EC',
 'electronics engineering' : 'EC',
 'biotechnology' : 'other',
 'other' : 'other',
 'industrial & production engineering' : 'other',
 'chemical engineering' : 'other',
 'applied electronics and instrumentation' : 'EC',
 'computer science and technology' : 'CS',
 'telecommunication engineering' : 'EC',
 'mechanical and automation' : 'ME',
 'automobile/automotive engineering' : 'ME',
 'instrumentation engineering' : 'EC',
 'mechatronics' : 'ME',
 'electronics and computer engineering' : 'CS',
 'aeronautical engineering' : 'ME',
 'computer science' : 'CS',
 'metallurgical engineering' : 'other',
 'biomedical engineering' : 'other',
 'industrial engineering' : 'other',
 'information & communication technology' : 'EC',
 'electrical and power engineering' : 'EL',
 'industrial & management engineering' : 'other',
 'computer networking' : 'CS',
 'embedded systems technology' : 'EC',
 'power systems and automation' : 'EL',
 'computer and communication engineering' : 'CS',
 'information science' : 'CS',
 'internal combustion engine' : 'ME',
 'ceramic engineering' : 'other',
 'mechanical & production engineering' : 'ME',
 'control and instrumentation engineering' : 'EC',
 'polymer technology' : 'other',
 'electronics' : 'EC'}


# In[59]:


df['Specialization'] = df['Specialization'].map(specialization_map)
df['Specialization'].unique()


# In[ ]:





# ## Column : 'Domain'

# In[60]:


df['Domain'].value_counts()


# ## This column has missing values in form of -1

# In[61]:


sns.boxplot(df['Domain'])
plt.show()


# In[62]:


## As we can see outlier,it is better to use median to replace the missing values.
df['Domain'].replace(-1,df['Domain'].median(),inplace=True)
df.head()


# # Univariate

# In[63]:


fig, axes = plt.subplots(6, 4, figsize=(18, 24))
axes = axes.flatten()

num_plots = min(len(col), len(axes))

for i, column in enumerate(col[:num_plots]):
    sns.histplot(data=df[column], ax=axes[i], kde=True, color='green', alpha=0.5)
    axes[i].set_title(column)  # Set subplot title
    axes[i].set_facecolor('white')
    axes[i].set_xlabel('')  # Remove x-axis label to improve clarity
    axes[i].set_ylabel('')  # Remove y-axis label to improve clarity

# Hide empty subplots
for ax in axes[num_plots:]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[64]:


fig, axes = plt.subplots(6, 4, figsize=(18, 24))
axes = axes.flatten()

# Ensure the number of columns does not exceed the number of subplots
num_plots = min(len(col), len(axes))

for i, column in enumerate(col[:num_plots]):
    axes[i].hist(df[column], bins=10, color='orange')  # Change the color to 'yellow'
    axes[i].set_title(column)

# Hide empty subplots
for ax in axes[num_plots:]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# In[65]:


fig, axes = plt.subplots(6, 4, figsize=(18, 24))
df_filled = df.fillna(-1)
axes = axes.flatten()

# Ensure number of plots doesn't exceed number of columns
num_plots = min(len(col), len(axes))

for i, column in enumerate(col[:num_plots]):
    axes[i].boxplot(df_filled[column])
    axes[i].set_title(column)

# Hide extra subplots
for ax in axes[num_plots:]:
    ax.axis('off')

plt.tight_layout()
plt.show()


# # Data Visualization

# In[66]:


sns.barplot(data=df,x='Specialization',y='Salary')


# - Aspirants from CE Branch earns the highest average pay
# - All the other branches earn nearly similar average pay

# In[ ]:





# In[67]:


# Define custom colors
colors = ['skyblue', 'salmon']


g = sns.FacetGrid(df, col="Gender", height=5)

# Map the distplot for each Gender category with the corresponding color
for i, gender in enumerate(df['Gender'].unique()):
    g.map(sns.distplot, "Salary", bins=50, color=colors[i], label=gender)


plt.legend()

plt.show()


# - We can observe that the salary data is right skewed.
# - We can also see that the distributions are quite similar for male and female in the range below .

# In[68]:


plt.figure(figsize=(25, 15))

# Set the plot style
sns.set_style("whitegrid")

# Use the custom color and plot
sns.countplot(data=df, y='CollegeState', palette='magma')

plt.yticks(fontsize=28)

plt.show()


# - Most of the aspirants are from Uttar Pradesh, followed by Karnataka, TamilNadu and Telangana.

# In[ ]:





# In[69]:


sns.kdeplot(x='Salary', data=df)
plt.title("KDE plot Salary")
plt.show()


# - The X-axis represents salaries ranging from 0 to 4, while the Y-axis represents densities ranging from 0 to 3 approximately. 
# - There is a sharp peak near zero on the Salary axis, indicating a significant number of individuals with low salaries.
# - As we move away from the peak, there is a gradual decrease in density, implying that fewer people earn higher salaries.

# In[ ]:





# In[70]:


plt.figure(figsize=(5, 5))


sns.set_style("darkgrid")


custom_palette = {'m': '#33bd1e', 'f': '#ad1382'}


sns.countplot(x='Gender', data=df, palette=custom_palette)

plt.show()

print(df['Gender'].value_counts())


# - The ratio of m/f is 3.19 indicates there are 3 times more men than women employed

# In[ ]:





# In[71]:


plt.figure(figsize=(5,5))
sns.boxplot(y='Salary', x='Gender', data=df)
plt.xticks(rotation=90)
plt.show()


# - It is noted that there are many outliers in the salary data
# - There is not much difference between median salary for both genders.
# - We can also observe male have more outliers indicating they are more people getting higher pays in male than female category

# In[ ]:





# In[72]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Salary',y='Specialization',data=df)
plt.suptitle('Salary levels by specialization')


# - The median salary for individuals across all specializations is nearly similar. 
# - However, it's evident that individuals with specializations in CS/EC tend to receive higher salaries compared to others.

# In[ ]:





# ## Designation

# In[73]:


df['Designation'].value_counts()


# In[74]:


popular_Designation = df['Designation'].value_counts()[:20].index.tolist()
print(popular_Designation)


# In[75]:


### Unique professions
top_Designations = df[df['Designation'].isin(popular_Designation)]
print(f"Unique professions : {len(df['Designation'].unique())}")
top_Designations.head()


# In[76]:


plt.figure(figsize=(20,10))

# Define custom colors
custom_palette = {'m': 'green', 'f': 'red'}

# Set the plot style
sns.set_style("whitegrid")

# Use the custom palette and plot
sns.countplot(x='Designation', hue='Gender', data=top_Designations, palette=custom_palette)

plt.xticks(fontsize=30, rotation=90)
plt.yticks(fontsize=30)

plt.show()


# - Across all professions, males dominate, as evidenced by a significant disparity in frequency for each role. 
# - The most common roles among Amcat aspirants, which predominantly fall within the domain of 'IT Roles.
# 

# In[ ]:





# In[77]:


specialization_counts = df['Specialization'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(specialization_counts, labels=specialization_counts.index, colors=plt.cm.viridis(np.linspace(0, 1, len(specialization_counts))))
plt.title("Specialization Distribution")
plt.show()


# In[78]:


plt.figure(figsize=(5,5))
sns.countplot(x='Specialization', data=df)
#plt.title("Sales By Region")
plt.xticks(rotation=90)
#plt.text(5,6,"Hello")
plt.show()


# - The majority of Amcat aspirants have specialized in Computer Science (CS) and Electronics and Communication (EC).

# In[79]:


plt.figure(figsize=(5,5))
sns.countplot(x='Degree', data=df, palette='viridis')
#plt.title("Sales By Region")
plt.xticks(rotation=90)
#plt.text(5,6,"Hello")
plt.show()


# - Most of the Amcat aspirants are from Btech domain.

# In[ ]:





# In[80]:


plt.figure(figsize=(20,10))
custom_palette = {'m': 'yellow', 'f': 'red'}
sns.barplot(x='Designation',y='Salary',hue='Gender',data=top_Designations, palette= custom_palette)
plt.xticks(fontsize=30,rotation=90)
plt.yticks(fontsize=30)
plt.show()


# - mean salary of top most frequent roles is nearly independent of gender.
# - there is some considerable difference in some roles.but we cannot be sure that women is being paid less in that role
# - it might be due to experience,specialization etc.

# In[81]:


### High paying designations and their relation with respect to gender
high = list(df.sort_values("Salary",ascending=False)["Designation"].unique())[:20]
high_pay = df[df['Designation'].isin(high)]
high_pay.head()


# In[82]:


plt.figure(figsize=(20,10))
custom_palette = {'m': 'cyan', 'f': 'magenta'}
sns.barplot(x='Designation',y='Salary',hue='Gender',data=high_pay, palette = custom_palette  )
plt.xticks(fontsize=30,rotation=90)
plt.yticks(fontsize=30)
plt.show()


# - Most of the high paying jobs are from IT domain.
# - In 45% of top paying roles,men are generally paid higher compared to women.
# - In 20% of top paying roles,women are paid higher than men
# - In roles like junior manager,sales account manager,software engineer trainee there are no women working in these fields.
# - Junior manager is highest paying for men and field engineer is the highest paying role for women.
# - The disperancy between pay based on gender might be because of other features like experience,specialization etc.
# - Software Enginner and Software developer are most frequent and highest paying jobs

# In[83]:


# Calculate average experience for each gender
mean_experience_f = df[df['Gender'] == 'f']['Experience'].mean()
mean_experience_m = df[df['Gender'] == 'm']['Experience'].mean()

# Plotting
plt.figure(figsize=(30,10))

sns.FacetGrid(df, hue="Gender", height=5) \
   .map(sns.distplot, "Experience") \
   .add_legend()

# Print average experience for each gender
print(f"Average experience (females): {mean_experience_f:.2f} years")
print(f"Average experience (males): {mean_experience_m:.2f} years")

plt.show()


# - It is a Bi-Modal Distribution
# - Average Experience of male is around 5.74 years and for women it is around 5.81 years.

# In[84]:


plt.figure(figsize= (7,7), dpi=100)
sns.heatmap(df[col].corr())


# - positive Correlations:
# - Higher college GPA tends to correlate positively with conscientiousness.
# - Stronger computer programming skills correlate with higher domain knowledge.
# - Negative Correlations:
# - Neuroticism negatively correlates with  conscientiousness.
# - Experience may have a negative correlation with age at graduation.
# 

# In[85]:


plt.figure(figsize=(55,15))

# Define custom palette
custom_palette = {'m': '#b6e314', 'f': '#14e3ce'}

# Use custom palette and pattern
sns.countplot(x="JobCity", data=high_pay, hue="Gender", palette=custom_palette)
plt.xticks(fontsize=38, rotation=90)
plt.yticks(fontsize=38)
plt.show()


# - Most amcat aspirants work in bengalore,Noida,Hyderabad and pune.
# - It is because most of them are from IT domain so majority of them work in these cities

# In[ ]:





# In[86]:


plt.figure(figsize=(10, 5))
sns.stripplot(data=high_pay, x='Experience', y='Salary', hue='Gender', jitter=True, palette='rocket')
plt.legend(title='Gender', loc='upper left')
plt.title('Salary Distribution by Experience and Gender (Strip Plot)')
#plt.xticks(rotation=45)
plt.show()


# - Median salary of both males and females have increased slightly with exp for first five years
# and Decreased suddenly on the 6th year and then same pattern for the following years.
# - We can see that men and women having same experience are paid nearly equally around 3.5-5 lakhs
# - we need to further see the distribution of experience with respect to Designation for both men and women
# to check whether women are being paid less in a role due to experience.

# In[ ]:





# In[87]:


plt.figure(figsize=(5,5))
g = sns.FacetGrid(data=high_pay, hue='Gender', height=8, palette='magma') \
    .map(sns.scatterplot, 'Experience', 'Salary') \
    .add_legend()
plt.show()


# - For same amount of experience,men are paid slightly higher than women in most of the cases.
# - Mostly men have highly paid jobs compared with women for every level of experience.

# In[ ]:





# In[88]:


### Now let us check relation with collegegpa
### first check the distribution of gpa 
plt.figure(figsize=(15,5))

# Create FacetGrid and map the distplot with green color
sns.FacetGrid(data=high_pay, col='Gender', height=5) \
   .map(sns.distplot, 'collegeGPA', color='green') \
   .add_legend()

plt.show()


# - We can see both distribution of college gpa vs male&female are normally distributed with mean around 75%.
# - We can see how Similar both male and female distributions are for collegeGPA
# - IQR is narrow indication most number of students have similar cgpa in 70-75% region.
# - There are some students with CGPA < 20 and CGPA>90

# In[ ]:





# In[89]:


plt.figure(figsize=(5, 5))

# Define custom palette
custom_palette = {'m': 'green', 'f': 'red'}

# Create FacetGrid and map the scatterplot with custom palette
sns.FacetGrid(data=high_pay, hue='Gender', height=8, palette=custom_palette) \
   .map(sns.scatterplot, 'collegeGPA', 'Salary') \
   .add_legend()

plt.show()


# - Most of men and women have cgpa around 70-75 which is a good range.
# - There is no relation that having high or low gpa being men/women is effecting the salary
# - so we can conclude this is not the reason for women being paid less as both distributions overlap extensively.

# In[ ]:





# In[90]:


plt.figure(figsize=(15, 5))

# Calculate Average Score and Academic Performance
df['AverageScore'] = (df['Logical'] + df['Quant'] + df['English']) / 3
df['Acadperf'] = (df['10percentage'] + df['12percentage'] + df['collegeGPA']) / 3

# Plotting the regression plots with color
plt.subplot(1, 2, 1)
sns.regplot(x='AverageScore', y='Salary', data=df, scatter_kws={"color": "#0b0491"})
plt.subplot(1, 2, 2)
sns.regplot(x='Acadperf', y='Salary', data=df, scatter_kws={"color": "#91044b"})
plt.show()


# - We can see there is some positive correlation of salary with both the Avgscore and Acadperformance.

# In[ ]:





# In[91]:


# For the total Dataset
## Checking whether specialization has any effect on salary
plt.figure(figsize=(20,10))
sns.barplot(data=df,x='Specialization',y='Salary',hue='Gender',palette='Set1')


# - Men from CS,EC,CE Earn slightly greater than women from this specialization.
# - Women from the EL specialization Earns way more than men from same specialization.

# In[92]:


# for the dataset containing Highpaying Jobs
plt.figure(figsize=(20,10))
sns.barplot(data=high_pay,x='Specialization',y='Salary',hue='Gender')


# - This is for the people who have higher pays.
# - From the CE Specialization,Only men are taking up higher pay jobs.
# - Mostly specialization is not the reason for women being paid less becuase as we say bulk of people are from cs and for cs men and women earn similar.

# In[ ]:





# ## Overall Insights from the Exploratory Data Analysis :
# - Most of Amcat Aspirants are male working in IT domain with an experience of around 5years with degree in B tech and specialization in Computer Science/Information Technology from tier-2 college in uttarpradesh with an average salary around 300k.
# - High paying jobs taken up by amcat aspirants are mostly from 'IT' Domain.
# - Software Engineer and Software Developer are the most aimed profession for amcat aspirants.

# In[ ]:





# ## Let us verify claims by Testing the Hypothesis

# ### Is there a relationship between gender and specialization? (i.e. Does the preference of Specialization depend on the Gender?)

# In[93]:


data=df[(df["Designation"].isin(["programmer analyst","software engineer","hardware engineer","associate engineer"])) & (df["Experience"]==0)]
plt.figure(figsize=(10,5))
sns.barplot(x="Designation",y="Salary",hue="Gender",data=data)


# - For Freshers,the salary is starting from 200000 and male are earing more than female as a fresher.

# ## 1 sample T-test to verify the claim
# - Null Hypothesis: mu = 275k
# - Alternate Hypothesis : mu != 275k

# In[94]:


# Import the t-test function from the scipy stats module
from scipy import stats as st

# Calculate the population mean
popmean = 250000 + 300000 / 2

# Perform the t-test and extract the p-value
pv = st.ttest_1samp(data['Salary'], popmean=popmean)[1]

# Set the significance level (alpha) for the test
alpha = 0.05

# Determine whether to reject or fail to reject the null hypothesis
if pv < alpha:
    print('We reject the null hypothesis and Average salary is not equal to 250k')
else:
    print('We fail to reject null hypothesis and Average salary is equal to 250k')


# In[ ]:





# ## Chi-Square Test to check the relation between Specialization and Gender.
# - Null Hypothesis: Gender does not impact specialization
# - Alternate Hypothesis : Gender impacts specialization

# In[95]:


# Import the chi-square test function from the scipy stats module
from scipy.stats import chi2_contingency as cst

# Create a contingency table for gender and specialization
sample_columns = pd.crosstab(df['Gender'], df['Specialization'], margins=True)

# Perform the chi-square test and extract the p-value
pv = cst(sample_columns)[1]

# Set the significance level (alpha) for the test
alpha = 0.05

# Determine whether to reject or fail to reject the null hypothesis
if pv < alpha:
    print('We reject the null hypothesis and Gender impacts specialization')
else:
    print('We fail to reject null hypothesis and Gender does not impact specialization')


# In[ ]:





# In[96]:


observed = pd.crosstab(df['Gender'], df['Specialization'])

# Plotting the stacked bar plot
observed.plot(kind='bar', stacked=True, figsize=(5, 5))
plt.title('Specialization Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotating x-axis labels for better readability
plt.legend(title='Specialization')
plt.show()


# - There are more men than woman in all the  Specialization categories.
# - Most of the Amcat Aspirants are from CS&EC specialization.

# ## So,The above made claims are True, the preference of Specialisation depends upon the Gender

# ## Bonus Question -
# ### What is the Average salary for each CollegeTier. if there is any relationship between CollegeTier and Salary.

# In[ ]:





# In[97]:


# Calculating average salary for each CollegeTier
average_salary_by_tier= df.groupby('CollegeTier')['Salary'].mean()

for tier, salary in average_salary_by_tier.items():
    print("Average Salary for Tier1 {}: {:.2f}".format(tier, salary))


# In[98]:


# Calculating average salary for each CollegeTier
average_salary_by_tier = df.groupby('CollegeTier')['Salary'].mean()

# Plotting
plt.figure(figsize=(5, 5))
average_salary_by_tier.plot(kind='bar', color='green')
plt.title('Average Salary by College Tier')
plt.xlabel('College Tier')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# - In the above plot we can see the Average salary earned by the College Tier 1 is Higher than the College Tier 2.
# - Average salary of Tier1 is 442356 and for Tier2 is 296893.

# ## Is there any relationship between CollegeTier and Salary.

# In[99]:


from scipy.stats import f_oneway

# Extract salaries for each college tier
college_tiers = df['CollegeTier'].unique()
salary_by_college_tier = [df[df['CollegeTier'] == tier]['Salary'] for tier in college_tiers]

# Perform ANOVA test
f_statistic, p_value = f_oneway(*salary_by_college_tier)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("There is a significant relationship between CollegeTier and Salary.")
else:
    print("Fail to reject the null hypothesis.")
    print("There is no significant relationship between CollegeTier and Salary.")


# In[100]:


### Lets us check salary with the College Tier
plt.figure(figsize=(5,5))
custom_palette = {'m': '#913104', 'f': '#197307'}
sns.barplot(data=high_pay,x='CollegeTier',y='Salary',hue='Gender', palette=custom_palette)


# In[101]:


df.groupby('CollegeTier').Gender.value_counts()


# In[102]:


data1 = df.groupby('CollegeTier').Gender.value_counts()


# In[103]:


# Grouping by CollegeTier and Gender and getting the counts
data1 =df.groupby('CollegeTier')['Gender'].value_counts()

# Plotting
plt.figure(figsize=(8, 6))
data1.unstack().plot(kind='bar', stacked=True)
plt.title('Gender Distribution by College Tier')
plt.xlabel('College Tier')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()


# - From the above bar graph,we can see that students from tier1 colleges have higher salary.
# - We can see more men are from tier1 colleges than women and overall most of the students are from tier2 colleges.
# - This might be a reason for women being paid less than men in high paying jobs because as most of women are from tier-2 colleges

# In[ ]:





# In[ ]:




