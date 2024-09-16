# This script gets numbers for the leaderboard page

# pip install pymysql
# pip install sqlalchemy

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, update, bindparam
import openai

#load envrionment variables
load_dotenv()
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY
engine = sqlalchemy.create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')
    
def get_database_connection():
    """
    This gets a connection to the HRF debate bot mysql database using SQLAlchemy
    Args:
        none

    Returns:
        sqlalchemy connection object

    """
    connection = engine.connect()
    return connection

def get_winner_counts():
    """
    This gets the counts of wins each candidate had. One win is one session where the
    user selected this candidate a majority of the time. Ties do not count as a win for anyone.

    Args:
        none

    Returns:
        dict {"ferguson_win_count": int, "reichert_win_count": int}

    """
    connection = get_database_connection()
    query = '''
    select r.userVoted,
        r.candidateId,
        q.sessionId
    from Response r join Query q on r.queryId = q.id 
    '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    votes_per_candidate_per_session = df.groupby(['sessionId', 'candidateId']).sum('userVoted').reset_index().pivot(index='sessionId', columns='candidateId', values='userVoted')
    print(votes_per_candidate_per_session)
    winner_per_session = votes_per_candidate_per_session
    winner_per_session['ferguson_winner'] = np.where(winner_per_session[1]>winner_per_session[2], 1, 0)
    winner_per_session['reichert_winner'] = np.where(winner_per_session[2]>winner_per_session[1], 1, 0)
    ferguson_win_count = winner_per_session['ferguson_winner'].sum()
    print("ferguson wins:", ferguson_win_count)
    reichert_win_count = winner_per_session['reichert_winner'].sum()
    print("reichert wins:", reichert_win_count)

    result = {"ferguson_win_count": int(ferguson_win_count), "reichert_win_count":int(reichert_win_count)}
    return result

# participant demographics, % party 

# % by age range

# % by gender

# top categories asked about

# 
def categorize_question(question):
    prompts=[{"role": "system", "content": 
        """
        You are a political scientist at a national newspaper. 
        Categorize the next statements to their most relevant political key issue.
        If the statement is not similar to any key issue, then categorize it as Other.
        The key issues are:
        Economy,
        Healthcare,
        Education,
        Immigration,
        Environment,
        National Security,
        Criminal Justice,
        Social Justice,
        Tax Policy,
        Gun Control,
        Infrastructure,
        Public Safety,
        Foreign Policy,
        Housing,
        Social Welfare Programs,
        Drug Policy,
        Veterans Affairs,
        Technology and Privacy,
        Election Integrity,
        Reproductive Rights,
        Gender,
        Religious Freedom
        
        """
        }]
    prompts.append({"role": "user", "content": question})
    answer = openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=20,
                messages=prompts,
                temperature=0
            )
    
    category = answer.choices[0].message.content.strip()
    return category

def categorize_all_questions():
    connection = get_database_connection()

    # get the questions from the db which have not been categorized yet
    query = '''
    select id as query_id, query
    from Query
    where category is null
    '''
    questions = pd.read_sql_query(query, connection)

    # categorize the questions
    questions['category'] = questions['query'].apply(categorize_question)
    df_dict = questions.to_dict('records')

    # update those rows with their category
    metadata = sqlalchemy.MetaData()
    query_table = sqlalchemy.Table('Query', metadata,
        Column("id", Integer, primary_key=True),
        Column("query", String),
        Column("category", String),
    )
    stmt = (
        update(query_table)
        .where(query_table.c.id == bindparam("query_id"))
        .values(category=bindparam("category"))
    )
    with connection.begin() as conn:
        conn.execute(
            stmt,
            df_dict,
        )
    print(stmt)

def get_top_categories(n):
    """
    This gets the names and counts of the top n most asked categories of questions.

    Args:
        n (int): the number of categories to get

    Returns:
        list of tuples, e.g. [("Economy", 9),
                                ("Healthcare", 8),
                                ("Education", 7),
                                ("Immigration", 6)]
    """
    connection = get_database_connection()
    query = '''
    select category, count(1) as n
    from Query
    group by category
    order by count(1) desc
    '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    top_n = df.head(n)
    top_categories = list(top_n.itertuples(index=False, name=None))
    print(top_categories)
    return top_categories


def get_participant_parties():
    """
    This gets the parties and percent of participants who said they were affiliated with each.

    Args:
        none

    Returns:
        dict, {"republican": int, "democrat": int, "other": int, "independent": int, "prefer_not_to_say": int}
    """
    connection = get_database_connection()
    query = '''
        select partyName, count(1) as n 
        from Session s join Party p on s.partyId = p.id
        group by partyName
        '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    total_count = df.n.sum()
    republican_count = df.loc[df['partyName'] == "Republican"]
    republican_count = republican_count.n.sum()
    republican_percent = republican_count/total_count

    democrat_count = df.loc[df['partyName'] == "Democrat"]
    democrat_count = democrat_count.n.sum()
    democrat_percent = democrat_count/total_count

    independent_count = df.loc[df['partyName'] == "Independent"]
    independent_count = independent_count.n.sum()
    independent_percent = independent_count/total_count

    other_count = df.loc[df['partyName'] == "Other"]
    other_count = other_count.n.sum()
    other_percent = other_count/total_count

    nosay_count = df.loc[df['partyName'] == "Prefer Not To Say"]
    nosay_count = nosay_count.n.sum()
    nosay_percent = nosay_count/total_count

    result = {"republican": republican_percent,
        "democrat": democrat_percent,
        "other": other_percent,
        "independent": independent_percent,
        "prefer_not_to_say": nosay_percent}
    print(result)
    return result


def get_participant_genders():
    """
    This gets the genders and percent of participants who said they identify with each.

    Args:
        none

    Returns:
        dict, {"male": int, "female": int, "nonbinary": int, "prefer_not_to_say": int}
    """
    connection = get_database_connection()
    query = '''
        select genderIdentity, count(1) as n 
        from Session
        group by genderIdentity
        '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    total_count = df.n.sum()
    male_count = df.loc[df['genderIdentity'] == "male"]
    male_count = male_count.n.sum()
    male_percent = male_count/total_count

    female_count = df.loc[df['genderIdentity'] == "female"]
    female_count = female_count.n.sum()
    female_percent = female_count/total_count

    nonbinary_count = df.loc[df['genderIdentity'] == "Non-Binary"]
    nonbinary_count = nonbinary_count.n.sum()
    nonbinary_percent = nonbinary_count/total_count

    nosay_count = df.loc[df['genderIdentity'] == "Prefer Not To Say"]
    nosay_count = nosay_count.n.sum()
    nosay_percent = nosay_count/total_count

    result = {"male": male_percent,
        "demofemalecrat": female_percent,
        "nonbinary": nonbinary_percent,
        "prefer_not_to_say": nosay_percent}
    print(result)
    return result

get_participant_genders()