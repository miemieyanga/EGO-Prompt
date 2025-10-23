
TASK_LABLES = {
    'swiss': ['<swissmetro>', '<car>', '<train>'],
    'trafficsafe': ['<no apparent injury>', '<minor injury>', '<serious injury>', '<fatal>'],
    'pandemic': ['<substantial decreasing>', '<moderate decreasing>', '<stable>', '<moderate increasing>', '<substantial increasing>']
}

TAGS = {
    'swiss': ['<Traveler Description>', '<\Traveler Description>'],
    'trafficsafe': ['<Crash Description>', '<\Crash Description>'],
    'pandemic': ['<Pandemic Description>', '<\Pandemic Description>']
}

Prompts = {
    "trafficsafe": {
        'CAUSAL_SYSTEM': """
        <SYSTEM PROMPT>
        Please generate a causal description based on the relations provided between <Causal Relations> and <\Causal Relations>, and the crash event details provided between <Crash Description> and <\Crash Description>.
        Your output will be used to support the reasoning process of a downstream model. Focus on delivering clear, structured causal reasoning grounded in the provided relations.
        Do **not** make direct severity predictions. Instead, describe relevant causal dynamics that may inform the likelihood of different crash severity outcomes, including: <no apparent injury>, <minor injury>, <serious injury>, and <fatal>.
        Don't only give conservetaive description, also consider <no apparent injury> and <fatal> situation.
        Where appropriate, incorporate probabilistic language (e.g., “likely,” “strongly associated,” “may contribute to”) to express the strength of each causal link. 
        Ensure that your reasoning is based solely on the available information and avoid introducing unsupported assumptions.
        <\SYSTEM PROMPT>

        <Causal Relations>
        1. [Person Status] affects [Severity]
         (The driver's Blood Alcohole Content - BAC will significantly increase the probability of fatal crashes. )
        2. [Position] affects [Severity]
        (Work zone can increase the probability of serious and fatal crashes. Driving in work zone after drinking is very likely to cause serious or fatal crashes.)
        3. [Driver Behavior] affects [Severity]
        (Aggressive driving and impairment-related behavior are more risk than other driver behaviors)
        <\Causal Relations>

        <Output>
        The output should follow:
        <Causal Description>
        Provide a numbered list of causal descriptions grounded in the provided causal relations and crash details. Each description should clearly explain the causal mechanism where applicable.
        <\Causal Description>
        <\Output>
        """,
        
        'CAUSAL_SYSTEM_CONSTRAINT': """
        Your revised prompt must follow the structure below:

        <SYSTEM PROMPT>
        (system prompt)
        <\SYSTEM PROMPT>
        <Causal Relations>
        (causal relations)
        <\Causal Relations>
        <Output>
        (output format, fixed, don't revise this)
        <\Output>

        **System Prompt Guidelines**  
        The system prompt should clearly instruct the model to generate a causal description based on the provided causal relations, with the goal of supporting the reasoning process of a downstream model.
        You should instruct model generate short, clear, and correct causal description.

        **Causal Relations Guidelines**  
        Only include causal relations between nodes for which corresponding information is available in the input description:  
        [Time], [Position], [Dynamic Conditions], [Infrastructure], [Road Conditions], [Road Level], [Driver Behavior], [Person Information], [Person Status], [Vehicle Status], [Vehicle Information], [Severity]

        <Operations>
        [1] **Add** new causal relations if they are clearly supported by the input. Do not make assumptions without evidence. Use the format:
        [Node A] affects [Node B]  
        (Explanation of how [Node A] affects [Node B])  
        Both [Node A] and [Node B] must come from the list above.  
        You may include node-to-node relations not involving the final prediction target to support broader reasoning and imputation.

        [2] **Modify** existing causal relations. You may:
        - Replace [Node A] affects [Node B] with a more accurate link such as [Node A] affects [Node C]  
        - Update the explanation for clarity or correctness

        [3] **Delete** any causal relation that is unsupported or may negatively impact model inference. Remove both the relation and its explanation.
        <\Operations>
        """,

        'SYSTEM': """
        Predict the crash severity reasoning on the causal descriptions provided between <Causal Description> and <\Causal Description>, and the crash event details provided between <Crash Description> and <\Crash Description>.
        Provide a single prediction enclosed in '<>' using one of the following labels: 
        <no apparent injury>, <minor injury>, <serious injury>, <fatal>.
        The last line of your response should only be of the following format: '<VALUE>' where VALUE is your prediction.
        """
    },
    'pandemic':{
        'CAUSAL_SYSTEM': """
        <SYSTEM PROMPT>
        Please generate a causal description based on the relations provided between <Causal Relations> and <\Causal Relations>, and the pandemic details provided between <Pandemic Description> and <\Pandemic Description>.
        Your output will be used to support the reasoning process of a downstream model. Focus on providing clear, structured causal reasoning grounded in the given relations.
        Do **not** make direct predictions about pandemic trends. Instead, describe the relevant causal dynamics that may inform the likely direction of hospitalization trends for the upcoming week. Possible trend categories include:  
        <substantial decreasing>, <moderate decreasing>, <stable>, <moderate increasing>, and <substantial increasing>.

        Definitions:
        - "Substantial" refers to changes greater than 3.
        - "Moderate" corresponds to changes between 1 and 3.
        - "Stable" is defined as changes between -1 and 1.

        Ensure all reasoning is based strictly on the provided information, and avoid making unsupported assumptions.
        Highlight key indicators that align with trend categories and emphasize the importance of recognizing stability and low volatility in data trends.
        <\SYSTEM PROMPT>

        <Causal Relations>
        1. [Demographic Information] affects [Vaccination Coverage] and [Restriction Policy Response]  
        Older or vulnerable populations often have higher vaccination uptake and are more likely to be targeted by stricter restrictions.
        2. [Healthcare System Condition] affects [Vaccination Coverage] and [Population Immunity]  
        Regions with better healthcare access can distribute vaccines more effectively and maintain higher baseline immunity.
        3. [ICU and Hospital Staffing Condition] affects [Restriction Policy Response]  
        When ICU beds are full or staffing is limited, governments tend to implement stricter control policies.
        4. [Vaccination Coverage] affects [Population Immunity]  
        Higher vaccination coverage directly increases the proportion of immune individuals in the population.
        5. [Population Immunity] affects [Reported Cases per 100k] and [Hospitalization per 100k]  
        Stronger immunity reduces both the number of new infections and the chance of severe cases needing hospitalization.
        6. [Reported Cases per 100k] affects [Hospitalization per 100k] and [Restriction Policy Response]  
        A rise in reported cases usually precedes more hospital admissions and can trigger policy tightening.
        7. [Hospitalization per 100k] affects [Restriction Policy Response]  
        High hospitalization levels often lead to immediate government intervention to limit further spread.
        8. [Hospitalization per 100k] and [Restriction Policy Response] affect [Change of Hospitalization Next Week]  
        The trends of hospitalization in past weeks have strong relation with change of hospitalization next week.
        <\Causal Relations>

        <Output>
        The output should follow:
        <Causal Description>
        Provide a numbered list of causal descriptions grounded in the provided causal relations and pandemic details. 
        Each description should clearly explain the causal mechanism.
        Use probabilistic language (e.g., “likely,” “strongly associated,” “may contribute to”) to express the strength or confidence of causal links where appropriate.  
        <\Causal Description>
        <\Output>
        """,
        
        'CAUSAL_SYSTEM_CONSTRAINT': """
        Your revised prompt must follow the structure below:

        <SYSTEM PROMPT>
        (system prompt)
        <\SYSTEM PROMPT>
        <Causal Relations>
        (causal relations)
        <\Causal Relations>
        <Output>
        (output format, fixed, don't revise this)
        <\Output>

        **System Prompt Guidelines**  
        The system prompt should clearly instruct the model to generate a causal description based on the provided causal relations, with the goal of supporting the reasoning process of a downstream model.
        Be clear, don't be too complex.

        **Causal Relations Guidelines**  
        Only include causal relations between nodes for which corresponding information is available in the input description:  
        [Demographic Information], [Healthcare System Condition], [ICU and Hospital Staffing Condition], [Vaccination Coverage], [Population Immunity],
        [Restriction Policy Response], [Hospitalization per 100k], [Reported Cases per 100k]

        <Operations>
        [1] **Add** new causal relations if they are clearly supported by the input. Do not make assumptions without evidence. Use the format:
        [Node A] affects [Node B]  
        (Explanation of how [Node A] affects [Node B])  
        Both [Node A] and [Node B] must come from the list above.  
        You may include node-to-node relations not involving the final prediction target to support broader reasoning and imputation.

        [2] **Modify** existing causal relations. You may:
        - Replace [Node A] affects [Node B] with a more accurate link such as [Node A] affects [Node C]  
        - Update the explanation for clarity or correctness

        [3] **Delete** any causal relation that is unsupported or may negatively impact model inference. Remove both the relation and its explanation.
        <\Operations>
        """,

        'SYSTEM': """
        Predict the trend of hospitalizations for the next week based on the causal descriptions provided between <Causal Description> and <\Causal Description>, and the pandemic details provided between <Pandemic Description> and <\Pandemic Description>.
        Provide a single prediction enclosed in '<>' using one of the following labels: 
        <substantial decreasing>, <moderate decreasing>, <stable>, <moderate increasing>, and <substantial increasing>. 
        Definitions:
        - "Substantial" refers to changes greater than 3.
        - "Moderate" corresponds to changes between 1 and 3.
        - "Stable" is defined as changes between -1 and 1.
        The final line of your response must follow this format: <VALUE>, where VALUE is your prediction.
        """
    },
    'swiss': {
        'CAUSAL_SYSTEM': """
        <SYSTEM PROMPT>
        Please generate a causal description based on the relations provided between <Causal Relations> and <\Causal Relations>, and the traveler details provided between <Traveler Description> and <\Traveler Description>.
        Your output will be used to support the reasoning process of a downstream model. Focus on providing clear, structured causal reasoning grounded in the given relations.
        Do **not** make direct predictions about travel mode choice. Instead, describe the relevant causal dynamics that may inform the likely travel mode choice. 
        Possible choices include:  <swissmetro>, <car>, <train>
        Use probabilistic language (e.g., “likely,” “strongly associated,” “may contribute to”) to express the strength or confidence of causal links where appropriate.  
        Ensure all reasoning is based strictly on the provided information, and avoid making unsupported assumptions.
        <\SYSTEM PROMPT>

        <Causal Relations>
        1. [Gender] and [age] affect [trip purpose] and [luggage]  
           (younger travelers are more likely to travel for education or leisure and carry luggage; older travelers more often travel for business with less luggage)
        2. [Income] affects [first class], [rail pass], and [self-paid]  
           (high-income travelers are more likely to choose first class, own a rail pass, and pay for the trip themselves)
        3. [Trip purpose] affects [self-paid] and [luggage]  
           (business trips are often employer-paid and involve less luggage; leisure trips are usually self-paid and involve more)
        4. [Origin and destination] determine [travel options], [travel time], and [headway]  
           (major city pairs offer more modes, shorter travel time, and higher frequency)
        5. [Trip purpose] affects [travel mode choice]  
           (business travelers tend to prefer faster, more reliable modes; leisure travelers may prioritize cost or flexibility)
        6. [First class] affects [travel mode choice]  
           (travelers choosing first class are more likely to select Train or Swissmetro over Car for comfort)
        7. [Rail pass] affects [travel mode choice]  
           (travelers with a rail pass are more likely to use Train or Swissmetro due to lower perceived cost)
        8. [Luggage] affects [travel mode choice]
           (travelers with heavy or bulky luggage may prefer Train or Car)
        9. [Trip_paid_by] affects [travel mode choice]
           (if the trip is employer-paid, travelers tend to choose faster or more comfortable modes like Swissmetro; if self-paid, they prefer cheaper options like standard Train or Car)
        10. [Travel time] and [headway] affect [travel mode choice]
            (business travelers are more sensitive to time and prefer faster and frequent modes; leisure travelers may tolerate longer travel time or wait if the mode is cheaper or more flexible)
        <\Causal Relations>

        <Output>
        The output should follow (enclosed in <Causal Description> and <\Causal Description>):

        <Causal Description>
        Provide a numbered list of causal descriptions grounded in the provided causal relations and traveler details. Each description should clearly explain the causal mechanism and, where applicable.
        <\Causal Description>

        <\Output>
        """,

        'CAUSAL_SYSTEM_CONSTRAINT': """
        Your revised prompt must follow the structure below:

        <SYSTEM PROMPT>
        (system prompt)
        <\SYSTEM PROMPT>
        <Causal Relations>
        (causal relations)
        <\Causal Relations>
        <Output>
        (output format, fixed, don't revise this)
        <\Output>

        **System Prompt Guidelines**  
        The system prompt should clearly instruct the model to generate a causal description based on the provided causal relations, with the goal of supporting the reasoning process of a downstream model.

        **Causal Relations Guidelines**  
        Only include causal relations between nodes for which corresponding information is available in the input description:  
        [gender], [age_range], [income], [first_class], [rail_pass], [trip_purpose], [luggage], [trip_paid_by], [origin_destination], 
        [options_count], [swissmetro_time_cost], [swissmetro_headway], [car_time_cost], [car_headway], [train_time_cost], [train_headway]

        <Operations>
        [1] **Add** new causal relations if they are clearly supported by the input. Do not make assumptions without evidence. Use the format:
        [Node A] affects [Node B]  
        (Explanation of how [Node A] affects [Node B])  
        Both [Node A] and [Node B] must come from the list above.  
        You may include node-to-node relations not involving the final prediction target to support broader reasoning and imputation.

        [2] **Modify** existing causal relations. You may:
        - Replace [Node A] affects [Node B] with a more accurate link such as [Node A] affects [Node C]  
        - Update the explanation for clarity or correctness

        [3] **Delete** any causal relation that is unsupported or may negatively impact model inference. Remove both the relation and its explanation.
        <\Operations>
        """,

        'SYSTEM': """
        Predict the travel mode choice reasoning on the causal descriptions provided between <Causal Description> and <\Causal Description>, and the traveler details provided between <Traveler Description> and <\Traveler Description>.
        Provide a single prediction enclosed in '<>' using one of the following labels: 
        <swissmetro>, <car>, <train>.
        The final line of your response must follow this format: <VALUE>, where VALUE is your prediction.
        """
    }
}