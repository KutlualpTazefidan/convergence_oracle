import requests
import json
import pandas as pd
import time
import string
import json
import psycopg2
import os
import sqlalchemy
from dotenv import load_dotenv
load_dotenv()

def main_semantic_scholar_api(start_date, end_date, journal):
    #### YOU CAN CHANGE THESE FIELDS 
    # research_field = ['Computer Science'] # Research field(s) to query (as a list)
    initial_offset = 0  # Initial start offset
    number_of_publications_per_request = 100 # Dont put more than a 100
    max_retries = 100  # Maximum number of retries for failed requests  
    amount_of_data = 100  # Total amount of data to retrieve
    alphabet_letters = ['A','E','I','U','O'] #  list(string.ascii_uppercase) # Research field(s) to query (as a list)
    # Gentle fetching
    # sleep time is one second because you can request every second based on the Semantic Scholar Regulations
    sleep_duration = 1  # Sleep duration in seconds between requests
    retry_sleep_duration = 1  # Sleep duration in seconds between requests
    #### YOU CAN CHANGE THESE FIELDS 
    
    # Define the API endpoint
    api_url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    length_df = 0
    
    # Fetching Data
    year_list = list(range(start_date, end_date + 1))
    retries_retry_exhausted = False
    year_list.reverse()
    print(year_list)
    for j, year in enumerate(year_list):
        print("Year",year)
        
        # Create an empty DataFrame to store the merged data of the year
        dataframe_for_year = pd.DataFrame()
        print(dataframe_for_year)
        for k,letter in enumerate(alphabet_letters):
            # First request to get the number of total papers to limit the loop
            params = {
                'query': alphabet_letters[k],  # Use the first research field from the list
                'venue': journal,
                'year': year,
                'limit': 1,  # Number of results per request
                'offset': 0,  # Offset for pagination
                'fields': 'title,authors,abstract,citationCount,year',
                # 'fields': 'externalIds,title,authors,abstract,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,citations,references,publicationVenue,year',
            }
            # Make the GET request
            response = requests.get(api_url, params=params,headers={'x-api-key':os.getenv("SEMANTICSCHOLAR_API_KEY")})
            # response = requests.get(api_url, params=params)
            # sleep time is one second because you can request every second based on the Semantic Scholar Regulations
            time.sleep(1)

            # Check if the request was successful
            if response.status_code == 200:
                data = json.loads(response.text)

            # calculate the batches required to get every publication for that topic
            num_batches, remainder = divmod(data['total'], number_of_publications_per_request)
            batches = [number_of_publications_per_request] * num_batches
            if remainder > 0:
                batches.append(remainder)
                
            print("For the letter: ", letter ," batches: ",batches)
            
            for i,batch in enumerate(batches):
                offset = initial_offset + number_of_publications_per_request * i  # Calculate the current offset
                retries = 0
                print("Letter: ", alphabet_letters[k]," Batch: ",str(batches[i]), " Iteration: ", i)
                while retries < max_retries:
                    # Define query parameters
                    params = {
                        'query': alphabet_letters[k],  # Use the first research field from the list
                        'venue': journal[0],
                        'year': year,
                        'limit': batch,  # Number of results per request
                        'offset': offset,  # Offset for pagination
                        # 'fields': 'externalIds,title,authors,abstract,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,citations,references,publicationVenue,year',
                        'fields': 'externalIds,title,abstract,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,publicationVenue,year',
                    }
                    print(response.text)

                    # Make the GET request
                    response = requests.get(api_url, params=params,headers={'x-api-key':os.getenv("SEMANTICSCHOLAR_API_KEY")})
                    # response = requests.get(api_url, params=params)
                    # Check if the request was successful
                    if response.status_code == 200:
                        data = json.loads(response.text)

                        # Create a DataFrame for unfiltered results
                        df = pd.DataFrame(data['data'])

                        break  # Successful request, exit the retry loop
                    else:
                        print(f"Error (Attempt {retries + 1}):", response.status_code)
                        retries += 1
                        if retries < max_retries:
                            print("Retrying after sleep: "+str(retry_sleep_duration*1)+"min")
                            time.sleep(retry_sleep_duration*1)
                        if retries == max_retries:
                            print("Fetch Failed")
                            retries_retry_exhausted = True
                            break
                    if retries_retry_exhausted == True:
                        print("Fetch Failed")
                        break
                if retries_retry_exhausted == True:
                    print("Fetch Failed")
                    break
                # print("total amount",response.json())
                length_df += len(df)  # Update the total length of unfiltered data

                # Save each iteration into a separate file - to keep some of the data in case of an error
                # The filename format includes the research field, offset, and is converted to lowercase
                
                # The saving is maybe not necc.
                #df.to_csv('../data/'+ str(year) +"_"+ journal[0]+ "_" + alphabet_letters[k].replace(" ", "_").lower() + "_" + str(offset) + '.csv', index=False)
                dataframe_for_year = pd.concat([dataframe_for_year,df],ignore_index=True)
                
                print(dataframe_for_year.shape[0])

                
                # Sleep for a specified duration between requests
                time.sleep(sleep_duration)
        if retries_retry_exhausted == True:
            print("Fetch Failed")
            break
        
        # Calculate the number of rows dropped
        dropped_count = len(dataframe_for_year) - len(dataframe_for_year.drop_duplicates(subset='paperId', keep='first'))

        # Remove rows with duplicate 'externalIdsIDs' values
        merged_data = dataframe_for_year.drop_duplicates(subset='paperId', keep='first')

        # Calculate the number of rows remaining
        remaining_count = len(merged_data)

        # Print the counts
        print(f"Removed {dropped_count} rows. Rows remaining: {remaining_count}")
        # old CSV code
        # merged_data.to_csv('../data/'+ str(year) +"_"+ journal[0]+"_"+str(remaining_count)+'no_cit_no_ref_no_auth.csv', index=False)
        ## connect to the SQL database and store the file there
        # change the data to string first
        merged_data_as_str = merged_data.astype(str)
        # function exists below
        dataframe_to_local_sql(merged_data_as_str, str(year), journal)
        time.sleep(sleep_duration)
    
    # Print the total length of unfiltered data after all iterations
    print("Length of the data", length_df)
    
def dataframe_to_local_sql(infun_dataframe, infun_year, infun_journal):
    # create the table name
    table_name = infun_journal + '_' + infun_year
    #read the database string from the .env
    DB_STRING = os.getenv('DB_STRING')
    # Defining the Engine
    engine = sqlalchemy.create_engine(DB_STRING)

    with engine.connect() as connection:
      infun_dataframe.to_sql(table_name, connection, if_exists='replace', index=False)