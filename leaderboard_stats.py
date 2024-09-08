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
    """
    This gets a connection to the HRF debate bot mysql database using SQLAlchemy
    Args:
        none

    Returns:
        sqlalchemy connection object

    """
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')
    return engine

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
    votes_per_candidate_per_session = df.groupby(['sessionId', 'candidateId']).sum('userVoted').reset_index().pivot(index='sessionId', columns='candidateId', values='userVoted')
    print(votes_per_candidate_per_session)
    winner_per_session = votes_per_candidate_per_session
    winner_per_session['ferguson_winner'] = np.where(winner_per_session[1]>winner_per_session[2], 1, 0)
    winner_per_session['reichert_winner'] = np.where(winner_per_session[2]>winner_per_session[1], 1, 0)
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
