import requests
import pandas as pd
import numpy as np
import rtree
import geopandas as gpd

URL_BASE = "https://api.chicagohealthatlas.org/"
INDICATORS = [
'total-population',
'non-hispanic-african-american-or-black',
'non-hispanic-asian-or-pacific-islander',
'hispanic-or-latino',
'non-hispanic-white',
'single-parent-households',
'limited-english-proficiency',
'violent-crime']

# functions to get neighborhood characteristics
def load_community_data(hardship_filepath):
    '''
    main function to load and clean community data
    '''
    print('getting data from chicago health atlas...')
    com_area = get_community_areas()
    indicators = get_indicators(com_area, INDICATORS)

    print('loading economic hardship index...')
    hard = pd.read_csv(hardship_filepath)

    print('merging community data...')
    merged = pd.merge(com_area, hard, how='left', left_on='name', 
                      right_on='COMMUNITY AREA NAME')
    merged = pd.merge(merged, indicators, how='left', on='slug')
    return merged

def get_community_areas():
    '''
    get a list of community areas and their geo location
    '''
    url = URL_BASE + "api/v1/places"
    j = requests.get(url).json()
    com_area = pd.DataFrame(j['community_areas'])
    com_area = com_area.drop(columns=['id', 'geo_type'])  
    return com_area
    
def get_indicators(com_area, indicators):
    '''
    get indicator values for each community area
    '''
    communities = com_area['slug']
    df = pd.DataFrame(index=communities, columns=indicators)
    
    for com in communities:
        for ind in indicators:
            url = URL_BASE + '/api/v1/topic_info/{}/{}'.format(com, ind)
            j = requests.get(url).json()
            j = j['area_data']
            if j:
                if ind == 'total-population':
                    df.loc[com, ind] = j[0]['number']
                else:
                    df.loc[com, ind] = j[0]['weight_percent']
    df = manual_correct(df)
    return df

def manual_correct(df):
    '''
    manually fill in montclare's indicator values, which are shown 
    on website but not included in dataset
    '''
    df.loc['montclare', 'total-population'] = 12992
    df.loc['montclare', 'non-hispanic-african-american-or-black'] = 4.5
    df.loc['montclare', 'non-hispanic-asian-or-pacific-islander'] = 4.0
    df.loc['montclare', 'hispanic-or-latino'] = 58.4
    df.loc['montclare', 'non-hispanic-white'] = 31.0
    df.loc['montclare', 'single-parent-households'] = 9.6
    df.loc['montclare', 'limited-english-proficiency'] = 23.9
    return df


# functions to load and clean cps data
def get_counselors_by_school(payroll_filepath):
    '''
    get number of counselor, counselor type (full time vs part time), and 
    their full time equivalent annual salary by school
    '''
    payroll = pd.read_csv(payroll_filepath)
    sc = payroll[payroll['Job Title'].isin(['School Counselor', 
                                            'Part-Time School Counselor'])]
    counsel_pay = sc.groupby(['Dept ID', 'FTE']).agg(
                              {'FTE Annual Salary': 'median'})
    counsel_cnt = sc.groupby(['Dept ID', 'FTE']).size().to_frame().reset_index()
    counsel_cnt = counsel_cnt.rename(columns={0:'num_counselors'})
    # merge median salary and headcount
    counselors = counsel_cnt.merge(counsel_pay, on=['Dept ID', 'FTE'])
    # collapse multiple rows of the same school
    counselors['num_counsel_FT'] = np.where(counselors['FTE'] == 1.0, 
                               counselors['num_counselors'], 0)
    counselors['num_counsel_PT'] = np.where(counselors['FTE'] == 0.5, 
                                  counselors['num_counselors'], 0)
    # if a school hires both FT and PT counselor, keep the full time salary
    counselors = counselors.groupby('Dept ID').\
                 agg({'num_counsel_FT': 'sum',
                      'num_counsel_PT': 'sum',
                      'FTE Annual Salary': 'max'}).\
                 reset_index() 
    return counselors

def clean_school_profiles(schools_filepath, schools_geo_filepath):
    '''
    clean CPS school profiles and add in geo location.
    '''
    schools = pd.read_csv(schools_filepath)

    # normalize demographic variables
    demo_cnt_cols = ['Student_Count_Low_Income',
       'Student_Count_Special_Ed', 'Student_Count_English_Learners',
       'Student_Count_Black', 'Student_Count_Hispanic', 'Student_Count_White',
       'Student_Count_Asian', 'Student_Count_Native_American',
       'Student_Count_Other_Ethnicity', 'Student_Count_Asian_Pacific_Islander',
       'Student_Count_Multi', 'Student_Count_Hawaiian_Pacific_Islander',
       'Student_Count_Ethnicity_Not_Available']
    for col in demo_cnt_cols:
          demo_perc = 'perc_' + col
          schools[demo_perc] = np.divide(schools[col], 
                                         schools['Student_Count_Total'])
    schools.drop(columns=demo_cnt_cols, inplace=True)

    # find racial majority
    race_perc_cols = ['perc_Student_Count_Black',
       'perc_Student_Count_Hispanic', 'perc_Student_Count_White',
       'perc_Student_Count_Asian', 'perc_Student_Count_Native_American',
       'perc_Student_Count_Other_Ethnicity',
       'perc_Student_Count_Asian_Pacific_Islander', 'perc_Student_Count_Multi',
       'perc_Student_Count_Hawaiian_Pacific_Islander',
       'perc_Student_Count_Ethnicity_Not_Available']
    schools['race_majority'] = schools[race_perc_cols].idxmax(axis=1)
    schools['race_majority'] = schools['race_majority'].str.replace('.+_', '')
    
    # add school location
    schools = add_school_location(schools, schools_geo_filepath)
    return schools

def add_school_location(schools, schools_geo_filepath):
    schools_geo = gpd.read_file(schools_geo_filepath)
    schools_geo = schools_geo[['school_id', 'geometry']]
    schools_geo['school_id'] = schools_geo['school_id'].astype('int')
    schools = schools.merge(schools_geo, how='left', 
                            left_on='School_ID', 
                            right_on='school_id')
    schools.drop(columns='school_id', inplace=True)
    return schools