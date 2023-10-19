import pandas as pd
import os

inputpath=r'./data_raw'
outputpath=r'./data'
filelist=os.listdir(r'./data_raw')
for file in filelist:
    # 训练集
    df = pd.read_csv(os.path.join(inputpath, file)).sort_values('Resolved').reset_index(drop=True)
    df = df[:int(0.6 * len(df))]
    df_group = df.groupby(['Issue key', 'Description'])['patch_nodiff'].apply(
        lambda x: x.str.cat(sep='\n')).reset_index()
    df_group.insert(loc=3, column='commit_time', value='')
    df_group.insert(loc=4, column='files', value='')
    df_group.insert(loc=5, column='issue_desc', value='')
    df_group.insert(loc=6, column='created_at', value='')
    df_group.insert(loc=7, column='closed_at', value='')
    df_group.insert(loc=8, column='summary', value='')
    df_group.insert(loc=9, column='issue_id', value=df_group['Issue key'])
    commit_file = df_group[['Issue key', 'patch_nodiff', 'summary', 'commit_time', 'files']].rename(
        columns={'Issue key': 'commit_id', 'patch_nodiff': 'diff'})
    issue_file = df_group[['Issue key', 'issue_desc', 'Description', 'created_at', 'closed_at']].rename(
        columns={'Issue key': 'issue_id', 'Description': 'issue_comments'})
    link_file = df_group[['Issue key', 'issue_id']].rename(columns={'Issue key': 'commit_id'})
    if not os.path.exists(os.path.join(outputpath, file[:-11], 'train')):
        os.makedirs(os.path.join(outputpath, file[:-11], 'train'))
    commit_file.to_csv(os.path.join(outputpath, file[:-11], 'train', 'commit_file.csv'))
    issue_file.to_csv(os.path.join(outputpath, file[:-11], 'train', 'issue_file.csv'))
    link_file.to_csv(os.path.join(outputpath, file[:-11], 'train', 'link_file.csv'))

    # 验证集
    df = pd.read_csv(os.path.join(inputpath, file)).sort_values('Resolved').reset_index(drop=True)
    df = df[int(0.6 * len(df)):int(0.8 * len(df))]
    df_group = df.groupby(['Issue key', 'Description'])['patch_nodiff'].apply(
        lambda x: x.str.cat(sep='\n')).reset_index()
    df_group.insert(loc=3, column='commit_time', value='')
    df_group.insert(loc=4, column='files', value='')
    df_group.insert(loc=5, column='issue_desc', value='')
    df_group.insert(loc=6, column='created_at', value='')
    df_group.insert(loc=7, column='closed_at', value='')
    df_group.insert(loc=8, column='summary', value='')
    df_group.insert(loc=9, column='issue_id', value=df_group['Issue key'])
    commit_file = df_group[['Issue key', 'patch_nodiff', 'summary', 'commit_time', 'files']].rename(
        columns={'Issue key': 'commit_id', 'patch_nodiff': 'diff'})
    issue_file = df_group[['Issue key', 'issue_desc', 'Description', 'created_at', 'closed_at']].rename(
        columns={'Issue key': 'issue_id', 'Description': 'issue_comments'})
    link_file = df_group[['Issue key', 'issue_id']].rename(columns={'Issue key': 'commit_id'})
    if not os.path.exists(os.path.join(outputpath, file[:-11], 'valid')):
        os.makedirs(os.path.join(outputpath, file[:-11], 'valid'))
    commit_file.to_csv(os.path.join(outputpath, file[:-11], 'valid', 'commit_file.csv'))
    issue_file.to_csv(os.path.join(outputpath, file[:-11], 'valid', 'issue_file.csv'))
    link_file.to_csv(os.path.join(outputpath, file[:-11], 'valid', 'link_file.csv'))

    # 测试集
    df = pd.read_csv(os.path.join(inputpath, file)).sort_values('Resolved').reset_index(drop=True)
    length = int(len(df) * 0.8)
    data = df.loc[length - 1, 'Issue key']
    df = df.loc[length:, :]
    while (df.loc[length, 'Issue key'] == data):
        length += 1
    df = df.loc[length:, :].reset_index(drop=True)
    file_name = df["file_name"].tolist()
    file_dict = {}
    num = 0
    file_list = []
    for i in file_name:
        if file_dict.get(i, -1) == -1:
            file_dict[i] = num
            file_list.append(num)
            num += 1
        else:
            file_list.append(file_dict.get(i, -1))
    df['commit_id'] = file_list
    df.insert(loc=10, column='issue_desc', value='')
    df.insert(loc=11, column='created_at', value='')
    df.insert(loc=12, column='closed_at', value='')
    df.insert(loc=13, column='summary', value='')
    commit_file = df[['commit_id', 'patch_nodiff', 'summary', 'commit_time', 'file_name']].rename(
        columns={'patch_nodiff': 'diff', 'file_name': 'files'})
    issue_file = df[['Issue key', 'issue_desc', 'Description', 'created_at', 'closed_at']].rename(
        columns={'Issue key': 'issue_id', 'Description': 'issue_comments'}).drop_duplicates(subset=['issue_id'])
    link_file = df[['Issue key', 'commit_id']].rename(columns={'Issue key': 'issue_id'})
    if not os.path.exists(os.path.join(outputpath, file[:-11], 'test')):
        os.makedirs(os.path.join(outputpath, file[:-11], 'test'))
    commit_file.to_csv(os.path.join(outputpath, file[:-11], 'test', 'commit_file.csv'))
    issue_file.to_csv(os.path.join(outputpath, file[:-11], 'test', 'issue_file.csv'))
    link_file.to_csv(os.path.join(outputpath, file[:-11], 'test', 'link_file.csv'))