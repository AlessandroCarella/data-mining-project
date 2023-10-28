import pandas as pd

df = pd.read_csv('dataset (missing + split)/Dataset_nomissing_nofeatur_noutlier.csv')
file_name = 'dataset (missing + split)/Dataset_nomissing_nofeatur_noutlier_noinconsistencies.csv'

mistake2correctname = {
    '100% of Disin U - Remixes' : '100% of Dissin You Remixes',
    '100% of Disin You - Remixes' : '100% of Dissin You Remixes',
    '100 % of Disin U - Remixes' : '100% of Dissin You Remixes',
    '100% of Disin\' You': '100% of Dissin You',
    'Back On a Mission': 'Back On A Mission',
    'Voz d\'Amor' : 'Voz D\' Amor',
    'Front by Front' : 'Front By Front',
    'I\'m Lovin\' It Remixed': 'I\'m Lovin\' It Remixes',
    'J\'adore Hardcore' : 'J\'Adore Hardcore',
    'J\'adore Hardcore - Extended Mix': 'J\'Adore Hardcore - Extended Mix',
    'Music For a Big Night Out': 'Music for A Big Night Out'
}
df_copy = df.copy()

df_copy['name'] = df_copy['name'].map(mistake2correctname).fillna(df_copy['name'])
df_copy['album_name'] = df_copy['album_name'].map(mistake2correctname).fillna(df_copy['album_name'])

df_copy.to_csv(file_name, index = False)
