# CHAPTER 5
# CONCLUSION AND RECOMMENDATION

## 5.1 Conclusion

This study successfully developed and deployed a machine learning-based electricity consumption prediction system for dormitory environments using smart meter data from Zamboanga City, Philippines. Through systematic comparison of three machine learning algorithms—Random Forest, XGBoost, and Support Vector Machine—the research achieved significant findings that contribute to the field of energy forecasting and smart building management.

### 5.1.1 Achievement of Research Objectives

The study accomplished all stated objectives:

**1. Data Collection and Preprocessing**
A synthetic smart meter dataset was constructed using empirically-derived parameters including actual dormitory room measurements, manufacturer-specified appliance power ratings, and ZAMCELCO electricity rates. The dataset comprises 2,089 records spanning 1.45 months (March 1 - April 14, 2024) from 8 dormitory rooms across 3 buildings. Comprehensive preprocessing addressed missing values (2.5%), temporal feature extraction, and realistic imperfections such as sensor communication errors and power fluctuations.

**2. Feature Engineering and Optimization**
Through rigorous feature engineering, the study identified 22 optimal features across five categories: temporal patterns (4 features), appliance usage (9 features), room characteristics (4 features), historical consumption (1 feature), and anomaly detection (1 feature). The feature selection process involved removing Month and Season features due to insufficient data range (1.45 months), excluding environmental variables (Temperature, Humidity, Wind_Speed) due to data quality concerns, and eliminating Appliance_kWh_Active to prevent data leakage. This optimization improved model accuracy from 91.55% to 92.03%.

**3. Model Comparison and Evaluation**
Three machine learning algorithms were systematically compared using confusion matrix-based metrics:
- **Random Forest**: 92.03% accuracy, 0.88 recall, 0.82 precision, 0.85 F1-score (best overall)
- **XGBoost**: 91.23% accuracy, 0.94 recall, 0.77 precision, 0.85 F1-score (highest recall)
- **Support Vector Machine**: 80.38% accuracy, 0.74 recall, 0.60 precision, 0.66 F1-score (baseline)

Random Forest emerged as the optimal model, providing the best balance between precision and recall for practical deployment.

**4. System Implementation and Deployment**
A fully functional web application was developed using Flask (backend) and HTML/CSS/JavaScript (frontend), deployed on Render cloud platform with automatic GitHub integration. The system provides real-time electricity consumption predictions, classification status (Normal/High), estimated kWh consumption, and cost calculations based on ZAMCELCO rates (₱10.50/kWh). The interface supports 22 input features including appliance states, temporal parameters, room characteristics, and historical consumption data.

### 5.1.2 Key Findings

**1. Model Performance**
The Random Forest classifier achieved 92.03% accuracy, which aligns with published energy forecasting literature reporting 85-95% accuracy for similar applications. The high recall (88%) ensures that the system successfully identifies 88% of high consumption events, making it suitable for early warning applications. The model's performance validates the effectiveness of appliance-based features and temporal patterns for electricity consumption prediction.

**2. Feature Importance**
Feature importance analysis revealed that the top three appliance features account for 95% of predictive power:
- Air Conditioner: 56% importance (dominant predictor)
- Electric Kettle: 35% importance (secondary predictor)
- Rice Cooker: 4% importance (tertiary predictor)

This finding confirms that high-wattage appliances are the primary drivers of electricity consumption in dormitory environments, providing actionable insights for energy management strategies.

**3. Feature Selection Impact**
The removal of noisy features (Month, Season, environmental variables) improved model accuracy by 0.48 percentage points (from 91.55% to 92.03%), demonstrating that fewer high-quality features outperform larger feature sets containing irrelevant or low-quality variables. This validates the importance of rigorous feature engineering in machine learning applications.

**4. Practical Applicability**
The deployed system demonstrates practical viability for real-world applications:
- Fast inference time (<1 millisecond per prediction)
- User-friendly interface requiring no technical expertise
- Scalable architecture supporting database integration for larger deployments
- Cost-effective deployment on cloud infrastructure

### 5.1.3 Contributions to Knowledge

This research contributes to the field of energy forecasting and smart building management in several ways:

**1. Methodological Framework**
The study establishes a comprehensive methodology for developing electricity consumption prediction systems, from data construction using empirically-derived parameters to model deployment on cloud platforms. This framework can be replicated for similar applications in educational institutions and residential settings.

**2. Model Comparison Insights**
The systematic comparison of three algorithms using confusion matrix-based metrics provides evidence-based guidance for model selection in electricity forecasting applications. The finding that Random Forest provides optimal balance between precision and recall offers practical value for practitioners.

**3. Feature Engineering Best Practices**
The study demonstrates the importance of feature quality over quantity, showing that removing noisy features can improve model performance. The identification of appliance-based features as dominant predictors (95% importance) provides actionable insights for future research.

**4. Practical Implementation**
The development of a production-ready web application demonstrates the feasibility of deploying machine learning models for real-time electricity consumption prediction, bridging the gap between academic research and practical application.

### 5.1.4 Limitations

While the study achieved its objectives, several limitations should be acknowledged:

**1. Data Scope**
The dataset spans only 1.45 months (March-April 2024), limiting the ability to capture seasonal patterns and long-term consumption trends. The synthetic nature of the data, while based on empirically-derived parameters, may not fully represent all complexities of real-world smart meter deployments.

**2. Geographic Specificity**
The system is calibrated for Zamboanga City's tropical climate and ZAMCELCO's electricity rates. Deployment in different geographic regions would require recalibration of parameters and potentially different feature sets.

**3. Appliance Coverage**
The model considers 9 common appliances but may not account for all electrical devices present in modern dormitory rooms. Emerging technologies and changing appliance usage patterns may require model updates.

**4. User Behavior Assumptions**
The model assumes relatively stable user behavior patterns. Significant changes in occupant behavior, academic schedules, or lifestyle factors may affect prediction accuracy.

### 5.1.5 Final Remarks

This study successfully demonstrates that machine learning techniques, particularly Random Forest classification, can effectively predict electricity consumption in dormitory environments with 92.03% accuracy. The research validates the use of appliance-based features and temporal patterns as primary predictors, while highlighting the importance of rigorous feature engineering for optimal model performance. The deployed web application provides a practical tool for energy management, offering real-time predictions that can support proactive decision-making by students, dormitory administrators, and utility providers.

The achievement of 92.03% accuracy with 88% recall demonstrates that the system is suitable for deployment as an early warning tool, successfully identifying the majority of high consumption events while maintaining acceptable false positive rates. The methodology and findings contribute to the growing body of knowledge in smart building management and energy forecasting, providing a foundation for future research and practical applications in educational institutions and residential settings.

---

## 5.2 Recommendations

Based on the findings and limitations of this study, the following recommendations are proposed for future research and practical implementation:

### 5.2.1 For Future Research

**1. Extended Data Collection**
Future studies should collect smart meter data spanning at least 12 months to capture seasonal patterns, including variations in heating and cooling demand across different seasons. This would enable the inclusion of Month and Season features, potentially improving prediction accuracy and providing insights into long-term consumption trends.

**2. Real-World Data Validation**
While this study used synthetic data based on empirically-derived parameters, validation using actual smart meter data from ZAMCELCO or similar utility providers would strengthen the findings. Establishing formal data sharing agreements with utility companies would enable comparison between synthetic and real-world model performance.

**3. Deep Learning Approaches**
Investigate deep learning architectures such as Long Short-Term Memory (LSTM) networks and Transformer models for time-series prediction. These approaches may capture complex temporal dependencies and non-linear patterns that traditional machine learning algorithms cannot model effectively.

**4. Multi-Building Deployment**
Expand the study to include multiple dormitory buildings across different campuses to assess model generalizability and identify building-specific consumption patterns. This would provide insights into how architectural design, insulation quality, and location affect electricity consumption.

**5. Weather Integration**
Incorporate real-time weather data and forecasts to improve prediction accuracy, particularly for temperature-sensitive appliances like air conditioners. This would require integration with weather APIs and development of weather-aware feature engineering techniques.

**6. User Behavior Modeling**
Develop user-specific consumption profiles using clustering techniques to personalize predictions based on individual behavior patterns. This could improve accuracy by accounting for variations in lifestyle, study schedules, and appliance usage preferences.

**7. Anomaly Detection Enhancement**
Investigate advanced anomaly detection techniques such as Isolation Forest, One-Class SVM, or Autoencoders to improve identification of unusual consumption patterns. This could help detect equipment malfunctions, unauthorized appliance usage, or billing errors.

**8. Cost-Benefit Analysis**
Conduct comprehensive cost-benefit analysis of implementing the prediction system at scale, including potential energy savings, reduced peak demand charges, and improved grid stability. This would provide economic justification for utility companies and educational institutions.

### 5.2.2 For System Enhancement

**1. Database Integration**
Migrate from file-based storage (CSV, JSON) to a relational database (PostgreSQL or MySQL) to support larger-scale deployment with thousands of rooms. This would enable efficient data management, user authentication, and historical data analysis.

**2. Real-Time Data Ingestion**
Develop integration with actual smart meter hardware to enable real-time data streaming and continuous model updates. This would require MQTT or similar IoT protocols for communication with smart meters.

**3. Mobile Application**
Develop native mobile applications (iOS and Android) to provide students with convenient access to consumption predictions, historical data, and energy-saving recommendations. Push notifications could alert users to predicted high consumption events.

**4. Predictive Alerts**
Implement automated alert system that notifies users via email or SMS when high consumption is predicted, enabling proactive action to reduce energy usage. Alerts could include specific recommendations based on appliance usage patterns.

**5. Energy Saving Recommendations**
Integrate an intelligent recommendation engine that suggests specific actions to reduce consumption based on current appliance usage and historical patterns. For example, recommending optimal air conditioner temperature settings or identifying opportunities to shift usage to off-peak hours.

**6. Dashboard Analytics**
Develop comprehensive analytics dashboard for dormitory administrators showing aggregate consumption trends, peak usage times, building-level comparisons, and cost projections. This would support data-driven energy management decisions.

**7. Model Retraining Pipeline**
Implement automated model retraining pipeline that periodically updates the model with new data, ensuring prediction accuracy remains high as consumption patterns evolve over time.

**8. Multi-Language Support**
Add support for multiple languages (English, Filipino, Cebuano) to improve accessibility for diverse user populations in Philippine educational institutions.

### 5.2.3 For Practical Deployment

**1. Pilot Program**
Conduct a pilot deployment in one dormitory building with actual smart meters to validate system performance in real-world conditions. Collect user feedback and measure actual energy savings achieved through the prediction system.

**2. Stakeholder Engagement**
Engage with key stakeholders including students, dormitory administrators, ZAMCELCO representatives, and university facilities management to ensure the system meets practical needs and addresses real pain points.

**3. User Training**
Develop comprehensive user training materials including video tutorials, user manuals, and FAQ documentation to ensure effective system adoption. Conduct training sessions for dormitory staff and student representatives.

**4. Privacy and Security**
Implement robust data privacy and security measures including encryption, secure authentication, and compliance with data protection regulations. Ensure student consumption data is anonymized and protected.

**5. Integration with Existing Systems**
Explore integration with existing university information systems, billing systems, and building management systems to provide seamless data flow and unified energy management.

**6. Incentive Programs**
Develop incentive programs that reward students for reducing electricity consumption based on system predictions. This could include recognition programs, reduced electricity fees, or sustainability certificates.

**7. Sustainability Education**
Use the system as an educational tool to raise awareness about energy consumption, environmental impact, and sustainable living practices among students. Integrate with sustainability courses and campus green initiatives.

**8. Scalability Planning**
Develop a phased rollout plan for scaling the system from pilot deployment to campus-wide implementation, including infrastructure requirements, budget estimates, and timeline projections.

### 5.2.4 For Policy and Management

**1. Energy Management Policies**
Develop institutional policies that leverage prediction system insights to establish consumption targets, peak demand management strategies, and energy efficiency standards for dormitory facilities.

**2. Demand Response Programs**
Implement demand response programs that use consumption predictions to shift non-essential loads to off-peak hours, reducing peak demand charges and supporting grid stability.

**3. Building Design Standards**
Use insights from consumption patterns to inform future dormitory building design, including appliance selection, insulation requirements, and renewable energy integration.

**4. Utility Collaboration**
Establish formal partnerships with ZAMCELCO and other utility providers to share data, coordinate demand management initiatives, and explore time-of-use pricing programs.

### 5.2.5 For Academic Institutions

**1. Curriculum Integration**
Integrate the system and its underlying methodology into computer science, electrical engineering, and environmental science curricula as a practical case study in machine learning applications.

**2. Research Collaboration**
Foster collaboration between computer science, engineering, and environmental science departments to conduct interdisciplinary research on smart building management and energy efficiency.

**3. Student Projects**
Use the system as a platform for student capstone projects, allowing students to extend functionality, improve algorithms, or develop new features as part of their academic requirements.

**4. Open Source Contribution**
Consider open-sourcing the system code (with appropriate data anonymization) to enable other educational institutions to adopt and adapt the solution for their specific contexts.

---

## 5.3 Closing Statement

This research demonstrates that machine learning techniques can effectively predict electricity consumption in dormitory environments, achieving 92.03% accuracy with Random Forest classification. The study contributes a comprehensive methodology from data construction to cloud deployment, validates the importance of appliance-based features as primary predictors, and delivers a practical web application suitable for real-world deployment.

The findings confirm that high-wattage appliances (air conditioner, electric kettle, rice cooker) account for 95% of predictive power, providing actionable insights for energy management strategies. The optimization of features from 24 to 22, removing noisy variables, improved accuracy and demonstrates the value of rigorous feature engineering.

While limitations exist—including limited temporal scope (1.45 months) and synthetic data—the research establishes a solid foundation for future work. The recommendations outline clear pathways for extending the research through longer data collection periods, real-world validation, deep learning approaches, and enhanced system features.

The deployed system at https://smartmeter-forecast.onrender.com demonstrates the practical viability of machine learning for energy management in educational institutions. With 88% recall, the system successfully identifies the majority of high consumption events, making it suitable for early warning applications that enable proactive energy management.

As educational institutions worldwide seek to reduce energy costs and environmental impact, this research provides a replicable framework for developing intelligent energy management systems. The methodology, findings, and deployed application contribute to the growing field of smart building management, offering practical value for students, administrators, and utility providers in the Philippines and beyond.

The future of energy management lies in intelligent systems that predict, inform, and empower users to make sustainable choices. This research takes a meaningful step toward that future, demonstrating that with appropriate data, rigorous methodology, and practical implementation, machine learning can transform how we understand and manage electricity consumption in residential and educational settings.

---

**END OF CHAPTER 5**

---

## Summary of Key Points

### Conclusions:
✅ Achieved 92.03% accuracy with Random Forest (best model)
✅ 22 optimized features outperform 24 features with noise
✅ Top 3 appliances account for 95% of predictive power
✅ System deployed and accessible at smartmeter-forecast.onrender.com
✅ 88% recall suitable for early warning applications
✅ Methodology replicable for similar applications

### Recommendations:
✅ Collect 12+ months of data for seasonal patterns
✅ Validate with real ZAMCELCO smart meter data
✅ Explore deep learning (LSTM, Transformers)
✅ Integrate database for scalability
✅ Develop mobile applications
✅ Implement automated alerts and recommendations
✅ Conduct pilot program in actual dormitory
✅ Establish utility partnerships for demand response

### Impact:
✅ Contributes to smart building management knowledge
✅ Provides practical tool for energy management
✅ Demonstrates ML applicability in educational settings
✅ Offers framework for future research and deployment
✅ Supports sustainability goals in Philippine institutions
