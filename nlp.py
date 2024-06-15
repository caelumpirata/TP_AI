import dask.dataframe as dd
import pandas as pd
from langchain_groq import ChatCompletion


df = dd.read_csv("met.csv")
df = df.compute()

for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col])#date formats

df = df.reset_index(drop=True)#reset index
df.columns = df.columns.str.replace(' ', '_')#replace spaces in the columns
cols = df.columns
cols = ", ".join(cols)

def clean_the_response(response):
    if "```" in response:
        pattern = r'```(.*?)```'
        code = re.search(pattern, response, re.DOTALL)
        extracted_code = code.group(1)
        extracted_code = extracted_code.replace('python', '')
        return extracted_code
    else:
        return response

def create_plot(user_input,cols):
  prompt = 'Write code in Python using Plotly to address the following request: {} ' \
             'Use df that has the following columns: {}.' \
             'Do not use animation_group argument and return' \
             'only code with no import statements and the data' \
             'has been already loaded in a df variable'.format(user_input, cols)

  # load api key from secrets
  groq_api_key = "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"
                   

  completion = ChatCompletion.create(
        model="llama3-8b-8192",
        max_tokens=1500,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

  response = completion.choices[0].message['content'].strip()
  extracted_code = clean_the_response(response)
  exec(extracted_code)



User_query = "draw a clustered column chart" #@param {type:"string"}
create_plot(User_query,cols)