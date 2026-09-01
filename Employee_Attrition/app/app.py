import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use('Agg')

st.set_page_config(
    page_title ='Employee Attrition Predictor',
    page_icon = '',
    layout ='wide',
    initial_sidebar_state ='expanded')

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/attrition_pipe.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model,features
model, feature_names = load_artifacts()
st.title('Employee Attrition Prediction System')
st.markdown('Predicting who is likely to resign before they do.')

st.sidebar.header('Employee Profile')
st.sidebar.caption('Fill in the employee details')
with st.sidebar.expander('Personal Details', expanded = True):
    age = st.slider('Age', 18, 60, 30)
    gender = st.selectbox('Gender',['Male','Female'])
    marital_status = st.selectbox('Marital Status',['Single','Married','Divorced'])
    distance_home = st.slider('Distance From Home(miles)',1,29,5)
    
with st.sidebar.expander('Job Details',expanded=True):
    department     = st.selectbox('Department',
                        ['Research & Development', 'Sales', 'Human Resources'])
    job_role       = st.selectbox('Job Role', [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative',
        'Manager', 'Sales Representative', 'Research Director',
        'Human Resources'])
    job_level          = st.selectbox('Job Level (1=Entry, 5=Senior)', [1,2,3,4,5])
    job_involvement    = st.selectbox('Job Involvement',
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:'Low',2:'Medium',3:'High',4:'Very High'}[x])
    job_satisfaction   = st.selectbox('Job Satisfaction',
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:'Low',2:'Medium',3:'High',4:'Very High'}[x])
    overtime           = st.selectbox('Works Overtime?', ['No', 'Yes'])
    business_travel    = st.selectbox('Business Travel',
                            ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])

with st.sidebar.expander('💰 Compensation', expanded=True):
    monthly_income     = st.number_input('Monthly Income ($)', 1000, 20000, 5000, 500)
    percent_hike       = st.slider('Last Salary Hike (%)', 11, 25, 13)
    stock_option       = st.selectbox('Stock Option Level (0=None, 3=High)', [0,1,2,3])

with st.sidebar.expander('📈 Experience & Satisfaction', expanded=True):
    total_working_yrs  = st.slider('Total Working Years', 0, 40, 8)
    years_at_company   = st.slider('Years at Company', 0, 40, 5)
    years_in_role      = st.slider('Years in Current Role', 0, 18, 3)
    years_since_promo  = st.slider('Years Since Last Promotion', 0, 15, 1)
    years_with_mgr     = st.slider('Years With Current Manager', 0, 17, 3)
    num_companies      = st.slider('Number of Companies Worked', 0, 9, 2)
    training_times     = st.slider('Training Sessions Last Year', 0, 6, 3)
    env_satisfaction   = st.selectbox('Environment Satisfaction',
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:'Low',2:'Medium',3:'High',4:'Very High'}[x])
    rel_satisfaction   = st.selectbox('Relationship Satisfaction',
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:'Low',2:'Medium',3:'High',4:'Very High'}[x])
    work_life_balance  = st.selectbox('Work-Life Balance',
                            [1,2,3,4], index=2,
                            format_func=lambda x:{1:'Bad',2:'Good',3:'Better',4:'Best'}[x])

predict_btn = st.sidebar.button('🔍 Predict Attrition Risk', type='primary',
                                 use_container_width=True)

travel_enc = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}

row = {f: 0 for f in feature_names}


row.update({
    'Age'                      : age,
    'BusinessTravel'           : travel_enc[business_travel],
    'DailyRate'                : 800,
    'DistanceFromHome'         : distance_home,
    'Education'                : 3,
    'EnvironmentSatisfaction'  : env_satisfaction,
    'Gender'                   : 1 if gender == 'Male' else 0,
    'HourlyRate'               : 65,
    'JobInvolvement'           : job_involvement,
    'JobLevel'                 : job_level,
    'JobSatisfaction'          : job_satisfaction,
    'MonthlyIncome'            : monthly_income,
    'MonthlyRate'              : 14000,
    'NumCompaniesWorked'       : num_companies,
    'OverTime'                 : 1 if overtime == 'Yes' else 0,
    'PercentSalaryHike'        : percent_hike,
    'PerformanceRating'        : 3,
    'RelationshipSatisfaction' : rel_satisfaction,
    'StockOptionLevel'         : stock_option,
    'TotalWorkingYears'        : total_working_yrs,
    'TrainingTimesLastYear'    : training_times,
    'WorkLifeBalance'          : work_life_balance,
    'YearsAtCompany'           : years_at_company,
    'YearsInCurrentRole'       : years_in_role,
    'YearsSinceLastPromotion'  : years_since_promo,
    'YearsWithCurrManager'     : years_with_mgr,
    
    'LogMonthlyIncome'         : np.log1p(monthly_income),
    'YearsPerCompany'          : total_working_yrs / max(num_companies, 1),
    'PromotionLag'             : years_since_promo - years_in_role,
    'IncomePerYear'            : monthly_income / (total_working_yrs + 1),
})

# HighRiskProfile flag
is_single   = 1 if marital_status == 'Single' else 0
row['HighRiskProfile'] = int(
    age < 32 and is_single == 1 and travel_enc[business_travel] == 2
)

# One-hot: Department
if 'Department_Research & Development' in row:
    row['Department_Research & Development'] = 1 if department == 'Research & Development' else 0
if 'Department_Sales' in row:
    row['Department_Sales'] = 1 if department == 'Sales' else 0

# One-hot: MaritalStatus
if 'MaritalStatus_Married' in row:
    row['MaritalStatus_Married'] = 1 if marital_status == 'Married' else 0
if 'MaritalStatus_Single' in row:
    row['MaritalStatus_Single'] = 1 if marital_status == 'Single' else 0

# One-hot: JobRole
role_key = f'JobRole_{job_role}'
if role_key in row:
    row[role_key] = 1


if predict_btn:
    X_input  = pd.DataFrame([row])[feature_names]
    prob     = model.predict_proba(X_input)[0][1]
    flagged  = prob >= 0.5

    # Risk tier
    if   prob >= 0.70:          risk, emoji, color = 'CRITICAL', '🔴', 'error'
    elif prob >= 0.5:     risk, emoji, color = 'HIGH',     '🟠', 'error'
    elif prob >= 0.4: risk, emoji, color = 'MEDIUM',   '🟡', 'warning'
    else:                       risk, emoji, color = 'LOW',      '🟢', 'success'

    # Results banner
    st.subheader('📊 Prediction Results')
    r1, r2, r3 = st.columns(3)
    r1.metric('Attrition Probability', f'{prob:.1%}',
              delta=f'{(prob - 0.161):+.1%} vs baseline')
    r2.metric('Risk Level', f'{emoji} {risk}')
    r3.metric('Threshold', "0.5",
              help='Score above this = flagged for HR review')

    if flagged:
        st.error(
            f'⚠ ATTRITION RISK DETECTED '
            f'Score {prob:.1%} . '
            f'Recommend HR retention conversation.')
    else:
        st.success(
            f'**✅ LOW ATTRITION RISK**  '
            f'Score {prob:.1%} '
            f'Employee appears stable.')

    # Risk score bar
    prob=float(prob)
    st.progress(prob, text=f'Risk score: {prob:.1%}')

    st.divider()


    col_shap, col_factors = st.columns([1.4, 1])

    with col_shap:
        st.subheader('🧠 SHAP Explanation — Why this score?')
        try:
            import shap
            xgb_model =model.named_steps['xgb']
            X_scaled = model[:-1].transform(X_input)
            explainer = shap.TreeExplainer(xgb_model)
            sv = explainer.shap_values(
                pd.DataFrame(X_scaled, columns=feature_names))
            sv = sv[1] if isinstance(sv, list) else sv
            exp_val = (explainer.expected_value[1]
                       if hasattr(explainer.expected_value, '__len__')
                       else explainer.expected_value)
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv[0],
                    base_values=exp_val,
                    data=X_scaled[0],
                    feature_names=feature_names
                ),
                show=False, max_display=12
            )
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True)
            st.caption(
                '🔴 Red bars push towards **attrition**.  '
                '🔵 Blue bars push towards **staying**.')
        except ImportError:
            st.info('Install shap for waterfall explanations: pip install shap')
        except Exception as e:
            st.info(f'SHAP explanation error: {e}')

    with col_factors:
        st.subheader('⚡ Risk Factors Summary')

        
        flags = []
        if overtime == 'Yes':
            flags.append('🔴 Works overtime — 3× higher attrition risk')
        if marital_status == 'Single':
            flags.append('🟠 Single — more mobile, higher exit rate')
        if business_travel == 'Travel_Frequently':
            flags.append('🟠 Frequent travel — burnout risk')
        if job_satisfaction <= 2:
            flags.append('🔴 Low job satisfaction — key driver of exits')
        if job_level == 1:
            flags.append('🟠 Entry-level — highest attrition tier')
        if stock_option == 0:
            flags.append('🟡 No stock options — less financial lock-in')
        if age < 30:
            flags.append('🟡 Age < 30 — exploring career options')
        if years_since_promo >= 3 and years_in_role >= 3:
            flags.append('🟠 Promotion lag — career stagnation signal')
        if monthly_income < 3000:
            flags.append('🔴 Below-median income — compensation risk')
        if num_companies >= 4:
            flags.append('🟠 Worked 4+ companies — history of job-hopping')

        
        protects = []
        if stock_option >= 2:
            protects.append('🟢 Stock options — strong retention incentive')
        if years_at_company >= 8:
            protects.append('🟢 Long tenure — high loyalty')
        if work_life_balance >= 3:
            protects.append('🟢 Good work-life balance')
        if job_satisfaction >= 3:
            protects.append('🟢 High job satisfaction')
        if overtime == 'No' and job_level >= 3:
            protects.append('🟢 Senior + no overtime — low burnout risk')

        if flags:
            st.markdown('Risk signals:')
            for f in flags:
                st.markdown(f)
        if protects:
            st.markdown('Protective factors:')
            for p in protects:
                st.markdown(p)
        if not flags and not protects:
            st.markdown('No strong risk signals detected for this profile.')

        st.divider()

        # Recommended HR action
        st.subheader('📋 Recommended Action')
        if prob >= 0.70:
            st.error('Immediate action required.\n'
                     '- Schedule 1:1 with manager this week\n'
                     '- Review compensation vs market rate\n'
                     '- Explore internal mobility options\n'
                     '- Consider retention bonus if flight risk confirmed')
        elif prob >= 0.5:
            st.warning('Schedule proactive check-in.\n'
                       '- Quarterly career development conversation\n'
                       '- Review workload and overtime hours\n'
                       '- Discuss promotion timeline if overdue')
        else:
            st.success('Routine engagement only.\n'
                       '- Include in standard pulsesurvey\n'
                       '- Maintain normal check-in cadence')
 