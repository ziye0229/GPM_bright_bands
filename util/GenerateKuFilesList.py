Ku_files_list = []
with open('../download_lists/cp_GPM_2ADPR_download_1.txt', 'r') as DPR_files:
    for file in DPR_files:
        if '/gpmdata/2016/' in file:
            if '/01/radar' in file or '/15/radar' in file:
                file = file.replace('2A.GPM.DPR.V8-20180723', '2A.GPM.Ku.V9-20211125').replace('V06A', 'V07A')
                Ku_files_list.append(file)

with open('../download_lists/GPM_2AKu_download.txt', 'w') as Ku_files:
    Ku_files.writelines(Ku_files_list)
