import re
import pandas as pd

def loadChat(filename):
    """ Loads text file and converts into a dataframe

    Parameters:
    -----------
    filename: text file name
        
    Returns:
    --------
    df : pandas dataframe
    
    """
    chat = open(filename)
    chatText = chat.read()
    chat.close()

    lines = []

    for line in chatText.splitlines():
        if line.strip() != "":
            lines.append(line.strip())

    clean = []
    pattern = r'\[(.*?)\]'
    for line in lines:
        if re.match(pattern, line):
            clean.append(line)
    df = pd.DataFrame(clean)
    df = df.rename(columns = {0: 'chat'})
    df['Date'] = df['chat'].str.extract(r'\[(.*?)\]')
    df['chat'] = df['chat'].replace(regex = r'\[(.*?)\] ', value='')
    df['User'] = df['chat'].str.extract(r'(.*?):')
    df['chat'] = df['chat'].replace(regex = r'(.*?): ', value='')
    df['Date'] = df['Date'].str.replace(r'\,', '')
    
    #df[['date', 'time']] = df['time'].str.split(expand=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y %H:%M:%S")
    
    # Extact Day of the Week
    df['Hour'] = df.apply(lambda row: row.Date.hour, axis = 1)
    df['Day_of_Week'] = df.apply(lambda row: row.Date.dayofweek, axis = 1)
    
    # Change labels for anonymization 
    user_labels = {old: new for old, new in zip(sorted(df.User.unique()), ['Her', 'Me'])}
    df.User = df.User.map(user_labels)
    
    df['Message_Only_Text'] = df.apply(lambda row: re.sub(r'[^a-zA-Z ]+', '', 
                                                          row.chat.lower()), 
                                       axis = 1)

    return df
