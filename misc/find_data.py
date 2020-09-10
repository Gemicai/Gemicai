import os
import dicomo

modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
modality = modalities[1]
data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_destination = '/home/nheinen/tsclient/niekh/Desktop/zgt/utilities/examples/dicom/'+modality+'/'

for root, dirs, files in os.walk(data_origin):
    for file in files:
        try:
            d = dicomo.Dicomo(root + '/' + file)
            if d.modality == modality:
                dicomo.plot_dicomo(d)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
    print(root)
    if input('Want to continue looking? y/n') == 'n':
        break

print(data_destination)
