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

#load envrionment variables
load_dotenv()
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')


def get_database_connection():
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')
    return engine

def select_from_database(sql_string):
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute(sql_string)
    records = cursor.fetchall()
    connection.close()
    return records

#overall winner, one session counts as one vote
def get_winner_counts():
    connection = get_database_connection()
    query = '''
    select r.userVoted,
        r.candidateId,
        q.sessionId
    from Response r join Query q on r.queryId = q.id 
    '''
    df = pd.read_sql_query(query, connection)
    votes_per_candidate_per_session = df.groupby(['sessionId', 'candidateId']).sum('userVoted').reset_index().pivot(index='sessionId', columns='candidateId', values='userVoted')
    print(votes_per_candidate_per_session)
    winner_per_session = votes_per_candidate_per_session
    winner_per_session['ferguson_winner'] = np.where(winner_per_session[1]>winner_per_session[2], 1, 0)
    winner_per_session['reichert_winner'] = np.where(winner_per_session[2]>winner_per_session[1], 1, 0)
    #ties count toward no one
    ferguson_win_count = winner_per_session['ferguson_winner'].sum()
    print("ferguson wins:", ferguson_win_count)
    reichert_win_count = winner_per_session['reichert_winner'].sum()
    print("reichert wins:", reichert_win_count)

    result = {"ferguson_win_count": ferguson_win_count, "reichert_win_count":reichert_win_count}
    return result

# participant demographics, % party 

# % by age range

# % by gender

# top categories asked about

# 
